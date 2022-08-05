from typing import Optional, List

import torch
from torch import nn

from imfas.models.rank_transformer import PositionalEncoding
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import VariableSelectionNetwork, \
    GatedResidualNetwork


class HierarchicalTransformer(nn.Module):
    def __init__(self,
                 input_dim_meta_feat: int,
                 input_dim_algo_feat: int,
                 input_dim_lc: int,
                 d_model: int,
                 n_head: int,
                 dim_feedforward: int,
                 n_layers_lc: int,
                 n_layers_meta: int,
                 dropout: float,
                 output_dim: int = 1,
                 norm_first: bool = False,
                 readout=None,
                 device: torch.device = torch.device('cpu')
                 ):
        super(HierarchicalTransformer, self).__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.device = device
        self.readout = readout if readout is not None else nn.Linear(d_model, output_dim)

        self.project_layer_meta_feat = nn.Identity() \
            if input_dim_meta_feat == d_model else nn.Linear(input_dim_meta_feat, d_model)
        self.project_layer_algo_feat = nn.Identity() \
            if input_dim_algo_feat == d_model else nn.Linear(input_dim_algo_feat, d_model)
        self.project_layer_lc = nn.Identity() if input_dim_lc == d_model else nn.Linear(input_dim_lc, d_model)

        # Follow the implementation from pytroch-forecasting but we simplify the architectures. Given that hidden
        # embeddings might be requery by different models
        self.flattened_grn = GatedResidualNetwork(2 * d_model, 2, 2, dropout, False)
        self.softmax = nn.Softmax(dim=-1)

        self.positional_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_head,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True,
                                                   norm_first=norm_first)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   nhead=n_head,
                                                   dropout=dropout,
                                                   dim_feedforward=dim_feedforward,
                                                   batch_first=True,
                                                   )

        self.lc_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers_lc)
        self.meta_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers_meta)

        self.lc_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layers_lc)
        self.meta_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers_meta)

    def forward(self,
                X_lc: torch.Tensor,
                X_meta_features: torch.Tensor,
                tgt_algo_features: torch.Tensor,
                tgt_meta_features: torch.Tensor,
                query_algo_features: List[torch.Tensor],
                n_query_algo: torch.IntTensor,
                query_algo_lc: List[torch.Tensor],
                query_algo_padding_mask: List[torch.Tensor],
                tgt_algo_lc: torch.Tensor,
                tgt_algo_padding_mask: torch.Tensor
                ):
        """
        Forward pass to hierarchy Transformer. Only learning curves from each individual datasets ([L, N_features])
        are passed to the Transformer of the first stage. Then we take one token from each dataset to aggregate the
        information from different datasets and then pass them to the decoders
        TODO consider Attention across different datasets on each individual time steps.(transpose X_lc into
         [B, L, N_datasets, N_features])

        Args:
            X_lc: torch.Tensor,
                input learning curve tensors with shape [B, N_datasets, L, N_lcfeat] a 4D tensor representing the
                learning curve data from the target algorithm. Each element (B) in this tensor represent the learning
                curves ([L, N_features]) of one algorithm on multiple datasets (N_datasets)
            X_meta_features: torch.Tensor,
                meta features of meta datasets, a tensor with shape [B, N_datasets, N_metafeatures]
            tgt_algo_features: torch.Tensor,
                algorithms features, a tensor with shape [B, N_algofeature]
            tgt_meta_features : torch.Tensor,
                meta features of test datasets of shape [B, N_metafeatures].
            query_algo_features: List[torch.Tensor],
                query algorithm features, a list contain B tensors: each tensor is a 2D
                tensors: [N_algoquery, N_algofeature]
            n_query_algo: torch.IntTensor,
                number of algorithm configurations that are evaluated on the test set (which contain the learning
                curve information)
            query_algo_lc: Optional[List[torch.Tensor]]
                learning curves of query algorithms on the target datasets,a list of tensors. Each element in the list
                contains the learning curve of the query algorithms on the target dataset. The learning curves is a
                tensor of shape [n_algo_i, L_max_lc, N_lcfeat]. n_algo_i is the number of evaluated configurations on
                task i. L_max_lc is the maximal length of the learning curves evaluated that will be fed to the decoder
            query_algo_padding_mask: Optional[List[torch.Tensor]]
                padding masks of query algorithms on the target datasets to indicate the incomplete learning curves
            tgt_algo_lc: torch.Tensor,
                learning curve of the target HP setting on test set with shape [B, L_max_lc, N_features]
            tgt_algo_padding_mask: Optional[torch.Tensor]
                masks for target algorithms to indicate the incomplete learning curves
        """
        data_shape = X_lc.shape
        batch_size = data_shape[0]
        n_datasets_encoder = data_shape[1]
        lc_length = data_shape[2]
        n_lc_feat = data_shape[3]

        assert len(query_algo_features) == batch_size

        en_feat_embeddings = self.project_layer_meta_feat(X_meta_features)
        en_feat_embeddings = en_feat_embeddings.view(batch_size * n_datasets_encoder, -1)  # [B * N_datasets, d_model]

        en_algo_embeddings: torch.Tensor = self.project_layer_algo_feat(tgt_algo_features)  # [B, d_model]
        en_algo_embeddings_repeat = en_algo_embeddings.repeat(1, n_datasets_encoder)
        en_algo_embeddings_repeat = en_algo_embeddings_repeat.view(batch_size * n_datasets_encoder, -1)

        en_meta_embedding = self.select_variable(en_feat_embeddings,
                                                 en_algo_embeddings_repeat)  # [B * N_datasets,1, d_model]

        X_lc = self.project_layer_lc(X_lc.view(batch_size * n_datasets_encoder, lc_length, n_lc_feat))
        X_lc = self.positional_encoder(X_lc)  # [B * N_datasets,L, d_model]

        encoder_input = torch.cat([en_meta_embedding, X_lc], dim=1)

        # we only take the first item (meta-item) from each feature
        encoder_output = self.lc_encoder(encoder_input)
        encoder_output = encoder_output.view(batch_size, n_datasets_encoder, 1 + lc_length, -1)[:, :, 0, :]
        encoder_output = self.meta_encoder(encoder_output)  # [batch_size, n_datasets_encoder,d_model]

        de_feat_embedding = self.project_layer_meta_feat(tgt_meta_features)
        de_feat_embedding_repeat = [
            de_f_emb.repeat(n_algo, 1) for de_f_emb, n_algo in zip(de_feat_embedding, n_query_algo)
        ]
        de_feat_embedding_repeat = torch.cat(de_feat_embedding_repeat, dim=0)

        de_algo_embeddings = self.project_layer_algo_feat(torch.cat(query_algo_features, dim=0))

        de_meta_embedding = self.select_variable(de_feat_embedding_repeat,
                                                 de_algo_embeddings)  # [sum(n_algo_test_set),1, d_model]

        query_algo_lc = self.project_layer_lc(torch.cat(query_algo_lc, dim=0))
        query_algo_lc = self.positional_encoder(query_algo_lc)  # [sum(n_algo_test_set),L_de, d_model]

        query_algo_lc = torch.cat([de_meta_embedding, query_algo_lc], dim=1)

        tgt_meta_embedding = self.select_variable(de_feat_embedding, en_algo_embeddings)

        tgt_algo_lc = self.project_layer_lc(tgt_algo_lc)
        tgt_algo_lc = self.positional_encoder(tgt_algo_lc)  # [B, L_max_lc, d_model]

        tgt_algo_lc = torch.cat([tgt_meta_embedding, tgt_algo_lc], dim=1)

        decoder_input = torch.cat([query_algo_lc, tgt_algo_lc], dim=0)  # [sum(n_algo_test_set)+ B, L_max, d_model]
        tgt_padding_masking = torch.cat([*query_algo_padding_mask, tgt_algo_padding_mask], dim=0)
        tgt_padding_masking = torch.cat(
            [torch.zeros([len(tgt_padding_masking), 1], dtype=torch.bool), tgt_padding_masking], dim=1
        )
        encoder_output_repeat = [
            en_o.repeat(n_algo, 1, 1) for en_o, n_algo in zip(encoder_output, n_query_algo)
        ]
        encoder_output_repeat = torch.cat([*encoder_output_repeat, encoder_output], dim=0)  # query + targets

        # [sum(n_algo_test_set) + batch_size, d_model]
        decoder_out = self.lc_decoder(tgt=decoder_input,
                                      memory=encoder_output_repeat,
                                      tgt_key_padding_mask=tgt_padding_masking)[:, 0]

        decoder_out_query_algos = torch.split(decoder_out[:-batch_size], n_query_algo.tolist())
        decoder_out_test = decoder_out[-batch_size:]

        n_algo_max = torch.max(n_query_algo) + 1  # maximal number of algorithm + test algo
        # pads decoder output
        decoder_out_padded = decoder_out_test.new_full((batch_size, n_algo_max, self.d_model), fill_value=0)
        # Finally, we only read the output of the last embedding value. Thus, we pad the zero values at the beginning
        # of each sequence.
        for i, decoder_query_algos in enumerate(decoder_out_query_algos):
            decoder_out_padded[i, -n_query_algo[i] - 1: -1] = decoder_out_query_algos[i]
            decoder_out_padded[i, -1] = decoder_out_test[i]

        tgt_mask_decoder_meta = torch.arange(0, n_algo_max)
        tgt_mask_decoder_meta = tgt_mask_decoder_meta < (n_algo_max - n_query_algo.unsqueeze(-1) - 1)

        decoder_output = self.meta_decoder.forward(decoder_out_padded,
                                                   memory=encoder_output,
                                                   tgt_key_padding_mask=tgt_mask_decoder_meta)[:, -1, :]
        prediction = self.readout(decoder_output)
        return prediction

    def select_variable(self, feat_embedding: torch.Tensor, algo_embedding: torch.Tensor):
        combined_feat = torch.stack([feat_embedding, algo_embedding], dim=-1)

        sparse_weights = self.flattened_grn(torch.cat([feat_embedding, algo_embedding], dim=-1))
        sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)
        meta_embedding = torch.sum(sparse_weights * combined_feat, -1)

        meta_embedding = meta_embedding.view(-1, 1, self.d_model)
        return meta_embedding


