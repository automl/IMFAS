from typing import Tuple

import torch

import imfas.losses.ranking_loss
from imfas.models.hierarchical_transformer import HierarchicalTransformer


class Trainer_Hierarchical_Transformer:
    def __init__(self, model: HierarchicalTransformer, loss_fn, optimizer, test_lim=5):
        self.step = 0
        self.losses = {
            # 'ranking_loss': 0
        }

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.test_lim = test_lim
        # self.n_slices = self.model.n_fidelities
        self.model.to(self.model.device)

    def evaluate(self, test_dataloader):
        test_lstm_losses = []
        test_shared_losses = []

        for X, y in enumerate(test_dataloader):
            # TODO write the evaluation parts
            pass

    def step_next(self):
        self.step += 1

    def train(self, train_dataloader):
        for X, y in train_dataloader:
            X_lc = X['X_lc'].float()
            X_meta_features = X['X_meta_features'].float()
            tgt_algo_features = X['tgt_algo_features'].float()
            tgt_meta_features = X['tgt_meta_features'].float()

            query_algo_features = X['query_algo_features'].float()  # [batch_size, n_query_algo], n_algo_feat
            query_algo_lc = X['query_algo_lc'].float()

            tgt_algo_lc = y['tgt_algo_lc'].float()  # [batch_size, L, n_query_algo, 1]

            labels = tgt_algo_lc[:, -1]

            decoder_input_shape = query_algo_lc.shape

            batch_size = decoder_input_shape[0]
            lc_length = decoder_input_shape[1]
            n_query_algos_all = decoder_input_shape[2]

            # flatten the first two dimensions
            query_algo_lc = torch.transpose(query_algo_lc, 1, 2).reshape(batch_size * n_query_algos_all, lc_length, 1)
            query_algo_features = query_algo_features.reshape(batch_size * n_query_algos_all, -1)

            # for each query set, select exactly n_query_algos[i] learning curves
            n_query_algos = torch.randint(0, n_query_algos_all + 1, (batch_size,))
            query_idx = torch.cat([torch.randperm(n_query_algos_all) for _ in range(batch_size)])
            query_idx = query_idx < n_query_algos.repeat_interleave(n_query_algos_all)

            query_algo_features = query_algo_features[query_idx]  # [sum(n_query_algos), n_query_algo]
            query_algo_lc = query_algo_lc[query_idx]  # [sum(n_query_algos), L, 1]

            # randomly mask out the tail of each learning curves. each learning curve needs to have at least 1 time step
            # and could be completely evaluated
            # TODO consider more sophisticated approach: dynamically reducing the mask sizes...
            n_query_lc = len(query_algo_lc)

            query_algo_lc, query_algo_padding_mask = self.mask_learning_curves(query_algo_lc, lc_length=lc_length,
                                                                               lower=1, upper=lc_length + 1,
                                                                               n_lc=n_query_lc
                                                                               )

            n_query_algos_all_list = n_query_algos.tolist()
            # query_algo_features = torch.split(query_algo_features, n_query_algos_all_list)
            # query_algo_lc = torch.split(query_algo_lc, n_query_algos_all_list)
            # query_algo_padding_mask = torch.split(query_algo_padding_mask, n_query_algos_all_list)

            # same as above, mask the learning curve of the target algorithm. However, we allow zero evaluations while
            # the full fidelity value should not be presented here
            tgt_algo_lc, tgt_algo_padding_mask = self.mask_learning_curves(tgt_algo_lc, lc_length=lc_length,
                                                                           lower=0, upper=lc_length, n_lc=batch_size)

            self.optimizer.zero_grad()
            # Dataset meta features and final  slice labels

            device = self.model.device

            predict = self.model(X_lc.to(device), X_meta_features.to(device),
                                 tgt_algo_features=tgt_algo_features.to(device),
                                 tgt_meta_features=tgt_meta_features.to(device),
                                 query_algo_features=query_algo_features.to(device),
                                 n_query_algo=n_query_algos,
                                 query_algo_lc=query_algo_lc.to(device),
                                 query_algo_padding_mask=query_algo_padding_mask.to(device),
                                 tgt_algo_lc=tgt_algo_lc.to(device),
                                 tgt_algo_padding_mask=tgt_algo_padding_mask.to(device))

            lstm_loss = self.loss_fn(input=predict, target=labels.to(device))
            lstm_loss.backward()

            self.optimizer.step()

    def mask_learning_curves(self,
                             lc: torch.Tensor,
                             n_lc: int,
                             lc_length: int,
                             lower: int,
                             upper: int,) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        mask the learning curves with 0. The masked learning curve has length between [lower, uppper)
        Args:
            lc: torch.Tensor,
                learning curves
            n_lc: int
                number of learning curves
            lc_length: int,
                length of the learning curves
            lower: int,
                minimal length of the learning curves after the masking
            upper: int
                maximal length of the learning curves after masking
        Returns:
            masked_lc: torch.Tensor
                masked learning curves. Masked values are 0
            padding_mask: torch.BoolTensor
                a tensor indicating which learning curves are masked

        """
        n_reserved_lc = torch.randint(lower, upper, (n_lc, 1))
        padding_mask = torch.arange(0, lc_length) >= n_reserved_lc

        masked_lc = ~padding_mask.unsqueeze(-1) * lc

        return masked_lc, padding_mask


class Trainer_Hierarchical_TransformerRankingLoss(Trainer_Hierarchical_Transformer):
    """Hierarchical Transformer to be trained with Ranking Loss"""
    def __init__(self, model: HierarchicalTransformer, loss_fn, optimizer, test_lim=5):
        super(Trainer_Hierarchical_TransformerRankingLoss, self).__init__(model, loss_fn, optimizer, test_lim)
        assert isinstance(loss_fn, (imfas.losses.ranking_loss.SpearmanLoss,
                                    imfas.losses.ranking_loss.WeightedSpearman))

    def train(self, train_dataloader):
        for X, y in train_dataloader:
            X_lc = X['X_lc'].float()  # [batch_size, n_dataset, lc_length, n_algos]
            X_meta_features = X['X_meta_features'].float()  # [batch_size, n_dataset, n_meta_features]
            algo_features = X['algo_features'].float()   # [batch_size, n_algos, n_algo_features]
            tgt_meta_features = X['tgt_meta_features'].float()  # [batch_size, n_meta_features]
            y_lc = y['y_lc'].float()

            target = y_lc[:, -1, :]

            encoder_input_shape = X_lc.shape

            batch_size = encoder_input_shape[0]
            n_datasets = encoder_input_shape[1]
            lc_length = encoder_input_shape[2]
            n_algos = encoder_input_shape[3]

            # the data is sorted as follows:
            # b1a1-d1, b1a1-d2 ...  |  b1a2-dn, b1a3-dn, ... * b1a1-dn
            # b1a2-d1, b2a2-d2 ...  |  b1a1-dn, b1a3-dn, ... * b1a2-dn
            # ...
            # b2a1-d1, b2a1-d2 ...  |  b2a2-dm, b2a3-dm, ... * b2a1-dm
            # b2a2-d1, b2a2-d2 ...  |  b2a1-dm, b2a3-dm, ... * b2a2-dm

            # batch size should be n_algo *

            # flatten the first two dimensions
            X_lc = torch.permute(X_lc, (0, 3, 1, 2)).reshape([batch_size * n_algos, n_datasets, lc_length, 1])

            X_meta_features = X_meta_features.repeat_interleave(n_algos, dim=0).view(batch_size * n_algos,
                                                                                     n_datasets, -1)
            tgt_meta_features = tgt_meta_features.repeat_interleave(n_algos, dim=0)

            # tgt_algo_features = algo_features.repeat_interleave(n_datasets, dim=1)
            tgt_algo_features = algo_features.reshape([batch_size * n_algos, -1])
            # assert torch.all(X_meta_features[0] == X_meta_features[1])

            # assert torch.all(tgt_algo_features[0,0] == tgt_algo_features[0, 1])
            # assert torch.all(tgt_algo_features[0,0] == tgt_algo_features[n_algos, 1])
            # assert torch.all(tgt_meta_features[0] == tgt_meta_features[1])

            y_lc = torch.permute(y_lc, (0, 2, 1)).reshape([batch_size * n_algos, lc_length, 1])
            n_valid_algos = torch.randint(0, n_algos + 1, (batch_size,))
            valid_lc_idx = torch.cat([torch.randperm(n_algos) for _ in range(batch_size)])
            valid_lc_idx = valid_lc_idx < n_valid_algos.repeat_interleave(n_algos)

            y_lc = y_lc[valid_lc_idx]
            y_algo_features = algo_features.view(batch_size * n_algos, -1)[valid_lc_idx]

            y_lc, y_lc_padding_masks = self.mask_learning_curves(y_lc, n_lc=len(y_lc), lc_length=lc_length,
                                                                 lower=0, upper=lc_length + 1)

            n_valid_algos_list = n_valid_algos.tolist()
            y_lc = y_lc.split(n_valid_algos_list)
            y_lc_padding_masks = y_lc_padding_masks.split(n_valid_algos_list)
            y_algo_features = y_algo_features.split(n_valid_algos_list)

            query_algo_lc = []
            query_algo_padding_mask = []
            query_algo_features = []

            tgt_algo_lc = []
            tgt_algo_padding_mask = []

            valid_lc_idx = valid_lc_idx.view(batch_size, n_algos)

            for lc, mask, algo_feat, n_valid_algo, valid_idx in \
                    zip(y_lc, y_lc_padding_masks, y_algo_features, n_valid_algos, valid_lc_idx):
                lc_query_tgt = lc.repeat(n_algos, 1, 1)
                mask_query_tgt = mask.repeat(n_algos, 1)
                algo_feat_tgt_query = algo_feat.repeat(n_algos, 1)

                tgt_lc = torch.zeros([n_algos, lc_length, 1], dtype=torch.float)
                tgt_mask = torch.ones([n_algos, lc_length], dtype=torch.bool)

                valid_idx_value = torch.where(valid_idx)[0] * n_valid_algo + torch.arange(n_valid_algo)
                lc_from_query = torch.ones(len(lc_query_tgt), dtype=torch.bool)
                lc_from_query[valid_idx_value] = False

                query_algo_lc.append(lc_query_tgt[lc_from_query])
                query_algo_padding_mask.append(mask_query_tgt[lc_from_query])
                query_algo_features.append(algo_feat_tgt_query[lc_from_query])

                lc_from_tgt = ~lc_from_query
                tgt_lc[valid_idx] = lc_query_tgt[lc_from_tgt]
                tgt_mask[valid_idx] = mask_query_tgt[lc_from_tgt]

                tgt_algo_lc.append(tgt_lc)
                tgt_algo_padding_mask.append(tgt_mask)

            query_algo_lc = torch.cat(query_algo_lc, dim=0)
            query_algo_padding_mask = torch.cat(query_algo_padding_mask, dim=0)
            query_algo_features = torch.cat(query_algo_features, dim=0)
            tgt_algo_lc = torch.cat(tgt_algo_lc, dim=0)
            tgt_algo_padding_mask = torch.cat(tgt_algo_padding_mask, dim=0)

            n_query_algos = (n_valid_algos.view(-1, 1) - valid_lc_idx.int()).flatten()

            self.optimizer.zero_grad()
            # Dataset meta features and final  slice labels
            device = self.model.device

            predict = self.model(X_lc.to(device), X_meta_features.to(device),
                                 tgt_algo_features=tgt_algo_features.to(device),
                                 tgt_meta_features=tgt_meta_features.to(device),
                                 query_algo_features=query_algo_features.to(device),
                                 n_query_algo=n_query_algos,
                                 query_algo_lc=query_algo_lc.to(device),
                                 query_algo_padding_mask=query_algo_padding_mask.to(device), tgt_algo_lc=tgt_algo_lc.to(device),
                                 tgt_algo_padding_mask=tgt_algo_padding_mask.to(device))

            lstm_loss = self.loss_fn(input=predict.view(batch_size, n_algos), target=target)
            lstm_loss.backward()

            self.optimizer.step()