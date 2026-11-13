from src.data.feature_datasets import AllScAIPFeatureDataset

if __name__ == '__main__':
    dataset = AllScAIPFeatureDataset(
        root='data/omol25/val',
        force_reload=True
    )
