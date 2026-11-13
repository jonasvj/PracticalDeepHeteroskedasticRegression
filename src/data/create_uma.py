from src.data.feature_datasets import UMAFeatureDataset

if __name__ == '__main__':
    dataset = UMAFeatureDataset(
        root='data/omol25/val',
        force_reload=True
    )
