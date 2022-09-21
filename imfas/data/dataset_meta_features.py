import pandas as pd
import torch
from torch.utils.data import Dataset

from imfas.data.preprocessings.transformpipeline import TransformPipeline


class DatasetMetaFeatures(Dataset):
    def __init__(self, path, transforms: TransformPipeline = None, index_col=0, *args, **kwargs):
        self.path = path
        self.transform = transforms

        self.df = pd.read_csv(path, index_col=index_col, *args, **kwargs)
        self.names = list(self.df.index)

        if transforms is not None:
            if transforms.fitted:
                self.transformed_df = self.transform.transform(self.df)
            else:
                self.transform = self.transform.fit(self.df)
                self.transformed_df = self.transform.transform(self.df)

    def __len__(self):
        return len(self.transformed_df)

    def __getitem__(self, idx):
        """
        idx: is the dataset's index; i.e. the dataset that is to be queried.
        it is of length n_meta_features.
        """
        if not isinstance(self.transformed_df, torch.Tensor):
            raise ValueError(f"You are trying to index a {type(self.transformed_df)} not tensor!")
        return self.transformed_df[idx]

    @property
    def shape(self):
        return self.transformed_df.shape

    def __repr__(self):
        return f"DatasetMetaFeatures(path={self.path}) , " \
               f"shape={self.shape}"
