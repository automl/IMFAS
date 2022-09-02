import torch

from imfas.data.dataset_join.dataset_join_taskwiseranking import Dataset_Joint_TaskswiseRanking


class Dataset_Joint_Taskswise(Dataset_Joint_TaskswiseRanking):
    """
    A Dataset
    """

    def __getitem__(self, item):
        """
        This dataset returns the following items:
        X_lc: torch.Tensor
            learning curve of the algorithm configuration on all the meta datasets with shape [N_dataset, L, N_feature]
        X_meta_features: torch.Tensor
            meta features of the meta dataset with shape [N_dataset, N_metafeatures]
        algo_features: torch.Tensor
            algorithm features with shape [N]
        y_meta_features: torch.Tensor
            meta features of the test set with shape [N_dataset, N_metafeatures]
        y_lc: torch.Tensor
            learning curves on the test datasets with shape [N_algo, L, N_features]
        """
        if self.is_test_set:
            return super(Dataset_Joint_Taskswise, self).__getitem__(item)
        idx_dataset = item % self.num_datasets
        algo = item // self.num_datasets

        dataset_y = self.split[idx_dataset]
        dataset_X = self.split[torch.arange(len(self.split)) != idx_dataset]

        tgt_meta_features = self.meta_dataset.transformed_df[dataset_y]

        X_meta_features = self.meta_dataset.transformed_df[dataset_X]
        X_lc = self.lc.transformed_df[dataset_X][:, :, [algo]]

        query_algo_idx = torch.arange(self.num_algos) != algo

        tgt_algo_features = self.meta_algo.transformed_df[algo]
        query_algo_features = self.meta_algo.transformed_df[query_algo_idx]

        y_lc = self.lc.transformed_df[dataset_y]

        tgt_algo_lc = y_lc[:, [algo]]
        query_algo_lc = y_lc[:, query_algo_idx]

        X = {
            'X_lc': X_lc,
            'X_meta_features': X_meta_features,
            'tgt_algo_features': tgt_algo_features,
            'query_algo_features': query_algo_features,
            'tgt_meta_features': tgt_meta_features,
            'query_algo_lc': query_algo_lc
        }

        y = {'tgt_algo_lc': tgt_algo_lc}

        return X, y

    def __len__(self):
        return self.num_datasets if self.is_test_set else self.num_datasets * self.num_algos
