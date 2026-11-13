import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Literal
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset
from src.utils.feature_extractor import PredictUnitFeatureExtractor


class FeatureDataset(InMemoryDataset):
    def __init__(
        self,
        root='data/omol25/val',
        split: Literal[0, 1 ,2] = 0,
        model: Literal['uma', 'allscaip'] = 'uma',
        transform=None,
        pre_transform=None,
        pre_filter=None,
        process_batch_size: int = 128,
        subset_size: int = 30_000,
        **kwargs
    ):
        self.split = split
        self.model = model
        self.process_batch_size = process_batch_size
        self.subset_size = subset_size
        super().__init__(root, transform, pre_transform, pre_filter, **kwargs)
        self.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        files_1 = [f'data{n:04d}.aselmdb' for n in range(80)]
        files_2 = [f'data{n:04d}.aselmdb-lock' for n in range(80)]
        return files_1 + files_2 + ['metadata.npz']


    @property
    def processed_file_names(self):
        return [f'{self.model}_features_split_{self.split}.pt']


    def get_feature_extractor(self):
        raise NotImplementedError


    def get_features(self, feature_dict, idx_vector, idx):
        raise NotImplementedError


    def get_list_of_data(self, x, preds, features):
        # Construct a list of PyG Data objects
        data_list = []
        mean = preds['energy']

        for idx in range(x.num_graphs):
            x_i = x.get_example(idx)
            mean_i = mean[idx]
            features_i = self.get_features(features, x.batch, idx)

            data = Data(
                z=x_i.atomic_numbers.to(torch.int),
                pos=x_i.pos.to(torch.float),
                y=x_i.energy.to(torch.float),
                mean=mean_i.to(torch.float),
                features=features_i,
            )

            data = data.to('cpu')
            data_list.append(data)

        return data_list


    def process(self):
        from fairchem.core.datasets import AseDBDataset
        from fairchem.core.datasets.base_dataset import Subset
        from fairchem.core.datasets.collaters.simple_collater import data_list_collater

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dataset_config = {
            'src': os.path.join(self.root, 'raw'),
            'a2g_args': {
                'r_energy': True,
                'r_forces': False,
                'r_stress': False,
                'r_edges': False,
                'task_name': 'omol',
                'r_data_keys': ['spin', 'charge'],
                'max_neigh': None,
                'radius': 6.0,
            }
        }
        dataset = AseDBDataset(dataset_config)
        metadata = dataset._metadata

        subset_size = self.subset_size # 30,000 for uma and 20,000 for allscaip
        rng = np.random.default_rng(seed=0)
        perm = rng.permutation(len(dataset))

        # Ensure same splits across both models
        if self.model == 'uma':
            split_1, split_2, split_3 = (
                    # 0:30,000          # 30,000:60,000                  # 60,000:90,000
                perm[:subset_size], perm[subset_size:subset_size*2], perm[subset_size*2:subset_size*3]
            )
        elif self.model == 'allscaip':
            offset = 30_000
            split_1, split_2, split_3 = (
                    # 0:20,000          # 30,000:50,000                  # 60,000:80,000
                perm[:subset_size], perm[offset:offset+subset_size], perm[offset*2:offset*2+subset_size]
            )

        feature_extractor = self.get_feature_extractor(device)

        for split, split_indices in tqdm(enumerate([split_1, split_2, split_3])):
            subset = Subset(dataset, split_indices, metadata)
            assert len(subset) == subset_size

            dataloader = DataLoader(
                subset,
                batch_size=self.process_batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=data_list_collater,
            )

            data_list = []
            diffs = torch.tensor([])
            with torch.no_grad():
                pbar = tqdm(dataloader, leave=False)
                for batch in pbar:
                    batch = batch.to(device)
                    preds, features = feature_extractor.predict(batch)
                    batch_list = self.get_list_of_data(batch, preds, features)
                    data_list.extend(batch_list)

                    diffs = torch.cat([
                        diffs, torch.abs(batch.energy - preds['energy']).detach().cpu()
                    ])
                    pbar.set_description_str(f'MAE: {diffs.mean():.3f}')

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            path = os.path.join(
                self.processed_dir,
                f'{self.model}_features_split_{split}.pt'
            )
            self.save(data_list, path)
            print(f'Finished preparing: {path}')


class UMAFeatureDataset(FeatureDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            model='uma',
            process_batch_size=128,
            subset_size=30_000,
            **kwargs
        )


    def get_feature_extractor(self, device: Literal['cpu', 'cuda']):
        from fairchem.core import pretrained_mlip
        from fairchem.core.units.mlip_unit.api.inference import inference_settings_default

        # Fairchem pull request #1714 
        inf_settings = inference_settings_default()
        inf_settings.energy_only = True
        # UMA
        predictor = pretrained_mlip.get_predict_unit(
            "uma-s-1p1",
            inference_settings=inf_settings,
            device=device,
        )
        feature_extractor = PredictUnitFeatureExtractor(
            module_names_to_extract=[
                'module.backbone.blocks.0',
                'module.backbone.blocks.1',
                'module.backbone.blocks.2',
                'module.backbone.blocks.3',
                'module.output_heads.energyandforcehead.head.energy_block.0',
                'module.output_heads.energyandforcehead.head.energy_block.2',
                'module.output_heads.energyandforcehead.head.energy_block.4',
            ],
            capture_module_input=[False]*6 + [True],
            detach_features=True,
        )
        feature_extractor.wrap_model(predictor)

        return feature_extractor


    def get_features(self, feature_dict, idx_vector, idx):
        feature_dict_i = dict()

        for feature_name, feature in feature_dict.items():
            feature = torch.clone(feature).to(torch.float)

            if 'backbone' in feature_name:
                feature = feature.narrow(1, 0, 1).squeeze(1)
            
            feature_dict_i[feature_name] = feature[idx_vector == idx]

        return feature_dict_i


class AllScAIPFeatureDataset(FeatureDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            model='allscaip',
            process_batch_size=16,
            subset_size=20_000,
            **kwargs
        )


    def get_feature_extractor(self, device: Literal['cpu', 'cuda']):
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit.api.inference import inference_settings_default

        # Fairchem pull request #1714 
        inf_settings = inference_settings_default()
        inf_settings.energy_only = True
        # AllScAIP
        allscaip_checkpoint = 'checkpoints/omol_all_sm_NeAnNoSi_ft_40E20F_fixed.pt'
        calc = FAIRChemCalculator.from_model_checkpoint(
            allscaip_checkpoint,
            task_name='omol',
            inference_settings=inf_settings,
            device=device,
        )
        predictor = calc.predictor

        feature_extractor = PredictUnitFeatureExtractor(
            module_names_to_extract=[
                'module.backbone.transformer_blocks.0.node_ffn.ffn_norm',
                'module.backbone.transformer_blocks.1.node_ffn.ffn_norm',
                'module.backbone.transformer_blocks.2.node_ffn.ffn_norm',
                'module.backbone.transformer_blocks.3.node_ffn.ffn_norm',
                'module.backbone.transformer_blocks.4.node_ffn.ffn_norm',
                'module.backbone.transformer_blocks.5.node_ffn.ffn_norm',
                'module.output_heads.energyandforcehead.head.energy_ffn.0.0',
                'module.output_heads.energyandforcehead.head.energy_ffn.1.0',
            ],
            capture_module_input=[False]*7 + [True],
            detach_features=True,
        )
        feature_extractor.wrap_model(predictor)

        return feature_extractor


    def get_features(self, feature_dict, idx_vector, idx):
        feature_dict_i = dict()

        for feature_name, feature in feature_dict.items():
            feature = torch.clone(feature).to(torch.float)
            feature_dict_i[feature_name] = feature[idx_vector == idx]

        return feature_dict_i
