import torch
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
            n_reserved_lc_query = torch.randint(1, lc_length + 1, (n_query_lc, 1))
            query_algo_padding_mask = torch.arange(0, lc_length) >= n_reserved_lc_query

            query_algo_lc = ~query_algo_padding_mask.unsqueeze(-1) * query_algo_lc

            n_query_algos_all_list = n_query_algos.tolist()
            query_algo_features = torch.split(query_algo_features, n_query_algos_all_list)
            query_algo_lc = torch.split(query_algo_lc, n_query_algos_all_list)
            query_algo_padding_mask = torch.split(query_algo_padding_mask, n_query_algos_all_list)

            # same as above, mask the learning curve of the target algorithm. However, we allow zero evaluations while
            # the full fidelity value should not be presented here
            n_reserved_lc_target = torch.randint(0, lc_length, (batch_size, 1))
            tgt_algo_padding_mask = torch.arange(0, lc_length) >= n_reserved_lc_target
            tgt_algo_lc = ~tgt_algo_padding_mask.unsqueeze(-1) * tgt_algo_lc

            self.optimizer.zero_grad()
            # Dataset meta features and final  slice labels

            predict = self.model(X_lc, X_meta_features, tgt_algo_features=tgt_algo_features,
                                 tgt_meta_features=tgt_meta_features, query_algo_features=query_algo_features,
                                 n_query_algo=n_query_algos, query_algo_lc=query_algo_lc,
                                 query_algo_padding_mask=query_algo_padding_mask, tgt_algo_lc=tgt_algo_lc,
                                 tgt_algo_padding_mask=tgt_algo_padding_mask)

            lstm_loss = self.loss_fn(input=predict, target=labels)
            lstm_loss.backward()

            self.optimizer.step()
