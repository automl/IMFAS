import math
import numpy as np
import pandas as pd
import torch

from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from imfas.data.preprocessings.lc_slice import LC_TimeSlices
from imfas.data.preprocessings.transformpipeline import TransformPipeline

from typing import Optional

class Dataset_LC(Dataset):
    def __init__(self, path, transforms: TransformPipeline, metric: str = "None"):
        """
        :param metric: Lcbench needs another specifier to subset the dataset.
        """
        self.path = path
        self.df = pd.read_hdf(path, mode="r")
        self.metric = metric

        if metric is not "None":
            self.df = self.df.xs(key=metric)

        # consider: is this possible by read in? - to reduce memory overhead

        self.multidex = self.df.index
        self.transforms = transforms

        if transforms is not None:
            if transforms.fitted:
                self.transformed_df = self.transforms.transform(self.df)
            else:
                self.transforms = self.transforms.fit(self.df)
                self.transformed_df = self.transforms.transform(self.df)

    def __getitem__(self, item: int):
        """
        :param item: int. index of dataset to be queried
        :returns: tensor of shape (n_fidelities, n_algorithms)
        """
        # FIXME: this won't be applicable no more when transform has ToTensor
        # return single learning curve:
        # self.df.loc[item, :]  # item:tuple e.g. ('APSFailure', '0')

        return self.transformed_df[item]

    def __len__(self):
        # Dataset major; i.e. dataset dimension!
        return self.shape[0]

    @property
    def shape(self):
        return self.transformed_df.shape

    def __repr__(self):
        return f"DatasetLC(path={self.path}, metric={self.metric}) , " \
               f"shape={self.shape}"

    def plot_single_dataset_curves(self, idx: int, ax=None, major='D'):
        if major == 'D':
            lcs = self.transformed_df[idx, :, :]
        elif major == 'A':
            lcs = self.transformed_df[:, idx, :]

        pd.DataFrame(lcs.numpy().T).plot(legend=False, ax=ax)

    def plot_alldataset_curves(self, dataset_name, major='D', **kwargs):
        if major == 'D':
            n = len(self)
        elif major == 'A':
            n = self.shape[1]  # number of algorithms

        r = math.ceil(math.sqrt(n))
        fig, axs = plt.subplots(ncols=r,
                                nrows=r,
                                figsize=(r * 1.5, r * 1.5),
                                layout="constrained",
                                sharex=True, **kwargs)

        for i, ax in zip(range(n), axs.flatten()):
            self.plot_single_dataset_curves(i, ax, major=major)

        for ax in axs.flatten()[i:]:
            ax.set_visible(False)

        fig.suptitle(f"Dataset {dataset_name} - {major} major")


class DatasetLCSynthetic(Dataset_LC):
    def __init__(self, path, transforms: TransformPipeline, metric: str = "None"):
        """
        :param metric: Lcbench needs another specifier to subset the dataset.
        """
        self.path = path
        self.transformed_df = torch.from_numpy(np.load(path))
        self.metric = metric


class DatasetTaskSet(Dataset_LC):
    def __init__(   self, 
                    path, 
                    transforms: TransformPipeline, 
                    metric: str = "None", 
                    n_datasets: Optional[int] = 1000, 
                    n_algos: Optional[int] = 100, 
                    ctype: str = 'train'
                ):
        """
        :param metric: Lcbench needs another specifier to subset the dataset.
        """
        
        self.type_idx = {
            'train': 0,
            'valid': 2,
            'test' : 3,
        }
        
        self.path = path
        self.transforms = transforms
        
        raw_data = np.load(path).mean(axis=2)[:n_datasets,:n_algos,:,self.type_idx[ctype]]
        pre_processed = self._preprocess(raw_data)

        self.transformed_df = torch.from_numpy(pre_processed)
        
        self.metric = metric
        
    # TODO Make this is a preprocesing step instead of this hack
    def _preprocess(self, np_array, scale_factor=100):
        '''
        Pre-Processing the data:
            - Handle Nans and Infs:
            - Normalize Min-Max 
        
        '''
        
        for i in range(np_array.shape[0]):  # tasks
            for j in range(np_array.shape[1]):  # AAlgorithms
                
                
                
                # # Normalize -- Min-Max
                np_array[i,j,:] = (np_array[i,j,:] - np_array[i,j,:].min()) / (np_array[i,j,:].max() - np_array[i,j,:].min())  
                np_array[i,j,:] = 1 - np_array[i,j,:]
                    
                if scale_factor:    
                    np_array[i,j,:] = scale_factor * np_array[i,j,:]

                for k in range(np_array.shape[2]):
                    
                    # Handle Nans
                    if np.isnan(np_array[i,j,k]) or np.isinf(np_array[i,j,k]):
                        # If the data point is the first one, set it to 0   
                        if k==0:
                            np_array[i,j,k] = 0
                        
                        # If the last point, or if the next point is also nan, use the last point
                        elif k==np_array.shape[2]-1 or ( np.isnan(np_array[i,j,k+1]) or np.isinf(np_array[i,j,k+1]) ) :
                            np_array[i,j,k] = np_array[i,j,k-1]
                        
                        # Otherwise take the average of the previous andthe next points
                        else:
                            np_array[i,j,k] = 0.5 * ( np_array[i,j,k-1] + np_array[i,j,k+1] ) 
        return np_array
    
    def __getitem__(self, item: int):
        """
        :param item: int. index of dataset to be queried
        :returns: tensor of shape (n_fidelities, n_algorithms)
        """
        # FIXME: this won't be applicable no more when transform has ToTensor
        # return single learning curve:
        # self.df.loc[item, :]  # item:tuple e.g. ('APSFailure', '0')

        return self.transformed_df[item]
    
    
    def __len__(self):
        return self.shape[0]
    
    
    @property
    def shape(self):
        return self.transformed_df.shape
    

    def __repr__(self):
        return f"DatasetLC(path={self.path}, metric={self.metric}) , " \
               f"shape={self.shape}"


if __name__ == "__main__":
    import os

    current = os.getcwd().split("/")
    file = "~/PycharmProjects/AlgoSelectionMF/data/preprocessed/LCBench/logs.h5"
    lc = Dataset_LC(file, "Train/val_accuracy", None)

    # lc.df.loc[('APSFailure', '0'), :]

    # select a time slice
    # lc.df[51].unstack().T

    from imfas.data.preprocessings.table_transforms import ToTensor
    from imfas.data.preprocessings.transformpipeline import TransformPipeline

    pipe = TransformPipeline([LC_TimeSlices(slice=51), ToTensor()])
    lc = Dataset_LC(file, "Train/val_accuracy", pipe)

    # lc[('APSFailure', '0')]

    lc[0]
