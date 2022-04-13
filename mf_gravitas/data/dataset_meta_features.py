import pandas as pd
import torch
from torch.utils.data import Dataset

from mf_gravitas.data.preprocessings.transformpipeline import TransformPipeline


class DatasetMetaFeatures(Dataset):
    def __init__(self, path, transforms: TransformPipeline = None, *args, **kwargs):
        self.path = path
        self.transform = transforms

        self.df = pd.read_csv(path, *args, **kwargs)
        self.names = list(self.df.index)

        if transforms is not None:
            if transforms.fitted:
                self.transformed_df = self.transform.transform(self.df)
            else:
                self.transformed_df = self.transform.fit(self.df)

    def __len__(self):
        return len(self.transformed_df)

    def __getitem__(self, idx):
        if not isinstance(self.transformed_df, torch.Tensor):
            raise ValueError('You are trying to index a dataframe not tensor!')
        return self.transformed_df[idx]
