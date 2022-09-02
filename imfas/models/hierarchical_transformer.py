from typing import List, Union

import torch
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import GatedResidualNetwork
from torch import nn

from imfas.models.rank_transformer import PositionalEncoding


class HierarchicalTransformer(nn.Module):
    def __init__(
            self,
            device: torch.device,
            input_dim_meta_feat: int,  # FIXME: number of (classical) dataset meta features
            input_dim_algo_feat: int,  # FIXME: number of hyperparameters / algo meta features
            input_dim_lc: int,  # Fixme: number of n_fidelity levels?
            d_model: int,
            n_head: int,
            dim_feedforward: int,
            n_layers_lc: int,
            n_layers_meta: int,
            dropout: float,
            output_dim: int = 1,
            norm_first: bool = False,
            mask_uncorrelated_lcs: bool = False,
            readout=None,
            incorporate_tgt_length_to_encoder: bool = False,
            incorporate_tgt_meta_features_to_encoder: bool = False
    ):
        """
        Hierarchical Transformer. both encoder layers and decoder layers are composed of two parts:
        the first part only do attention for each individual learning curves. Then the second part do attention across
        the embedded information across different datasets/ algos combination
        Args:
            # FIXME: DOC
            mask_uncorrelated_lcs: bool,
                if we want to mask the uncorrelated learning curves between encoders and decoders. This could be set as
                True if we don't want the attention correlation on the learning curves that come from different
                algorithms and datasets # FIXME: what are uncorrelated learning curves here?
            incorporate_tgt_length_to_encoder: bool,
                if we want to incorporate the information of the target length to the encoder (to variable selection
                network)
            incorporate_tgt_meta_features_to_encoder: bool,
                if we want to incorporate the information of the target algorithms features to the encoder
                TODO There are two ways of implementing this:
                  the first one is to feed it to the variable selection network
                  the second one is to compute the difference of the embedded meta features between the target datasets
                  and the meta dataset
                  as a first step, we select the first approach
        """
        super(HierarchicalTransformer, self).__init__()

        # save arguments
        self.d_model = d_model
        self.output_dim = output_dim
        self.device = device
        self.mask_uncorrelated_lcs = mask_uncorrelated_lcs
        self.incorporate_tgt_length_to_encoder = incorporate_tgt_length_to_encoder
        self.incorporate_tgt_meta_features_to_encoder = incorporate_tgt_meta_features_to_encoder
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.n_layers_lc = n_layers_lc
        self.n_layers_meta = n_layers_meta
        self.dropout = dropout
        self.norm_first = norm_first
        self.input_dim_meta_feat = input_dim_meta_feat
        self.input_dim_algo_feat = input_dim_algo_feat
        self.input_dim_lc = input_dim_lc

        self.build_model(readout)

    def build_model(self, readout) -> None:

        # FIXME: what does this do?
        if readout is not None:
            self.readout = nn.Linear(self.d_model, self.output_dim)

        # TODO: make this design choice a subclass of HierarchicalTransformer?
        #  I would highly appreciate it. the forward call is a mess (partly because of all the ifs)
        # def build_model(self, readout):
        #     # do your if else here:
        #     super(self).build_model(readout)

        # FIXME: why is it sensible to check against d_model, we might want to embed them anyways!

        # MLP for dataset meta features
        if self.input_dim_meta_feat == self.d_model:
            self.project_layer_meta_feat = nn.Identity()
        else:
            self.project_layer_meta_feat = nn.Linear(self.input_dim_meta_feat, self.d_model)

        # MLP for algo meta features (HP)
        if self.input_dim_algo_feat == self.d_model:
            self.project_layer_algo_feat = nn.Identity()
        else:
            self.project_layer_algo_feat = nn.Linear(self.input_dim_algo_feat, self.d_model)

        # MLP for learning curves
        if self.input_dim_lc == self.d_model:
            self.project_layer_lc = nn.Identity()
        else:
            self.project_layer_lc = nn.Linear(self.input_dim_lc, self.d_model)

        # Follow the implementation from pytroch-forecasting but we simplify
        # the architectures. Given that hidden
        # embeddings might be requery by different models
        n_var_encoder = 2  # FIXME: magic number?
        if self.incorporate_tgt_length_to_encoder:
            self.project_layer_length = nn.Linear(1, self.d_model)
            n_var_encoder += 1
        if self.incorporate_tgt_meta_features_to_encoder:
            n_var_encoder += 1

        # ---------------------------------------------------------------------

        self.flattened_grn_encoder = GatedResidualNetwork(
            n_var_encoder * self.d_model,
            n_var_encoder,
            n_var_encoder,
            self.dropout, False
        )
        self.flattened_grn_decoder = GatedResidualNetwork(
            2 * self.d_model, 2, 2, self.dropout, False
        )
        self.softmax = nn.Softmax(dim=-1)

        self.positional_encoder = PositionalEncoding(d_model=self.d_model, dropout=self.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
            norm_first=self.norm_first
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dropout=self.dropout,
            dim_feedforward=self.dim_feedforward,
            batch_first=True,
        )

        self.lc_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers_lc)
        self.meta_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers_meta)

        self.lc_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer,
                                                num_layers=self.n_layers_lc)
        self.meta_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.n_layers_meta)

    def forward(  # Fixme @difan: this is a mess. split it into multiple sub-functions, whose
            # names and documentation convey the purpose of each op.
            self,
            X_lc: torch.Tensor,
            X_meta_features: torch.Tensor,
            tgt_algo_features: torch.Tensor,
            tgt_meta_features: torch.Tensor,
            query_algo_features: Union[List[torch.Tensor], torch.Tensor],
            n_query_algo: torch.IntTensor,
            query_algo_lc: Union[List[torch.Tensor], torch.Tensor],
            query_algo_padding_mask: Union[List[torch.Tensor], torch.Tensor],
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
            query_algo_lc: Union[List[torch.Tensor], torch.Tensor]
                learning curves of query algorithms on the target datasets,a list of tensors. Each element in the list
                contains the learning curve of the query algorithms on the target dataset. The learning curves is a
                tensor of shape [n_algo_i, L_max_lc, N_lcfeat]. n_algo_i is the number of evaluated configurations on
                task i. L_max_lc is the maximal length of the learning curves evaluated that will be fed to the decoder
            query_algo_padding_mask: Union[List[torch.Tensor], torch.Tensor]
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

        # if isinstance(query_algo_lc, list):
        #    assert len(query_algo_lc) == batch_size
        # else:
        #    assert len(query_algo_lc) == sum(n_query_algo)

        # Encode dataset meta features
        en_feat_embeddings = self.project_layer_meta_feat(X_meta_features)
        en_feat_embeddings = en_feat_embeddings.view(  # [B * N_datasets, d_model]
            batch_size * n_datasets_encoder, -1
        )

        # Encode algorithm features:
        # for the first layer, we only compute the attention map within each learning curve
        # a1d1, a1d2, ... | a2dn, a3dn, ... * a1dn
        # a2d1, a2d2, ... | a1dm, a3dm, ... * a2dm
        en_algo_embeddings: torch.Tensor = self.project_layer_algo_feat(
            tgt_algo_features)  # [B, d_model]
        en_algo_embeddings_repeat = en_algo_embeddings.repeat(1, n_datasets_encoder)
        en_algo_embeddings_repeat = en_algo_embeddings_repeat.view(
            batch_size * n_datasets_encoder, -1
        )

        # assert torch.all(en_algo_embeddings_repeat[0] == en_algo_embeddings_repeat[1])
        # assert torch.any(en_algo_embeddings_repeat[0] != en_algo_embeddings_repeat[n_datasets_encoder])

        de_feat_embedding = self.project_layer_meta_feat(tgt_meta_features)

        # Merge both (dataset & algo meta features) into one embedding:
        encoder_meta_variables = [en_feat_embeddings, en_algo_embeddings_repeat]
        # FIXME: what is the purpose here: -------------------------------------
        #  optional (design choice): encode the length? WHY?
        if self.incorporate_tgt_meta_features_to_encoder:
            encoder_meta_variables.append(de_feat_embedding.repeat(n_datasets_encoder, 1))
        if self.incorporate_tgt_length_to_encoder:
            tgt_lc_length = torch.sum(~tgt_algo_padding_mask, dim=1, keepdim=True)
            tgt_lc_length /= lc_length  # to scale them back to [0,1]

            tgt_lc_length_embedding = self.project_layer_length(tgt_lc_length)
            encoder_meta_variables.append(tgt_lc_length_embedding.repeat(n_datasets_encoder, 1))
        # ----------------------------------------------------------------------

        # This is a feature softmax * vector selector.
        # Fixme: makes only sense, if the features are not encoded (identiy)!
        en_meta_embedding = self.select_variable(  # [B * N_datasets,1, d_model]
            encoder_meta_variables,
            self.flattened_grn_encoder
        )

        # Learning curve projection (using linear/identity layer)
        # FIXME: design choice!
        X_lc = self.project_layer_lc(
            X_lc.view(batch_size * n_datasets_encoder, lc_length, n_lc_feat)
        )
        # positional encoding of the embedded learning curve
        X_lc = self.positional_encoder(X_lc)  # [B * N_datasets, L, d_model]

        # "token": (embedded meta features + positional encoding of the learning curve)
        # token means this is our contextualized learning curve
        encoder_input = torch.cat([en_meta_embedding, X_lc], dim=1)

        # we only take the first item (meta-item) from each feature
        encoder_output = self.lc_encoder(encoder_input)
        encoder_output = encoder_output.view(
            batch_size, n_datasets_encoder, 1 + lc_length, -1
        )[:, :, 0, :]  # FIXME: comment?
        encoder_output = self.meta_encoder(
            encoder_output)  # [batch_size, n_datasets_encoder, d_model]

        de_feat_embedding_repeat = [
            de_f_emb.repeat(n_algo, 1) for de_f_emb, n_algo in zip(de_feat_embedding, n_query_algo)
        ]

        de_feat_embedding_repeat = torch.cat(de_feat_embedding_repeat, dim=0)

        # assert torch.all(de_feat_embedding_repeat[0] == de_feat_embedding_repeat[1])
        # assert torch.any(de_feat_embedding_repeat[0] != en_algo_embeddings_repeat[n_query_algo[0]])

        if isinstance(query_algo_features, list):
            query_algo_features = torch.cat(query_algo_features, dim=0)

        de_algo_embeddings = self.project_layer_algo_feat(query_algo_features)

        de_meta_embedding = self.select_variable(
            [de_feat_embedding_repeat, de_algo_embeddings],
            self.flattened_grn_decoder
        )  # [sum(n_algo_test_set),1, d_model]

        if isinstance(query_algo_lc, list):
            query_algo_lc = torch.cat(query_algo_lc, dim=0)

        query_algo_lc = self.project_layer_lc(query_algo_lc)
        query_algo_lc = self.positional_encoder(
            query_algo_lc)  # [sum(n_algo_test_set),L_de, d_model]

        query_algo_lc = torch.cat([de_meta_embedding, query_algo_lc], dim=1)

        tgt_meta_embedding = self.select_variable(
            [de_feat_embedding, en_algo_embeddings],
            self.flattened_grn_decoder
        )

        tgt_algo_lc = self.project_layer_lc(tgt_algo_lc)
        tgt_algo_lc = self.positional_encoder(tgt_algo_lc)  # [B, L_max_lc, d_model]

        tgt_algo_lc = torch.cat([tgt_meta_embedding, tgt_algo_lc], dim=1)

        decoder_input = torch.cat([query_algo_lc, tgt_algo_lc],
                                  dim=0)  # [sum(n_algo_test_set)+ B, L_max, d_model]

        # # fixme: tgt_padding_masking is used for what?
        if isinstance(query_algo_padding_mask, list):
            tgt_padding_masking = torch.cat([*query_algo_padding_mask, tgt_algo_padding_mask],
                                            dim=0)
        else:
            tgt_padding_masking = torch.cat([query_algo_padding_mask, tgt_algo_padding_mask], dim=0)

        tgt_padding_masking = torch.cat(
            [torch.zeros([len(tgt_padding_masking), 1], dtype=torch.bool, device=self.device),
             tgt_padding_masking], dim=1
        )

        # FIXME: Subclass?
        if self.mask_uncorrelated_lcs:
            n_query_algos_total = torch.sum(n_query_algo)
            encoder_output_repeat = encoder_output.new_full(
                (n_query_algos_total, *encoder_output.shape[1:]),
                fill_value=0)
            encoder_output_repeat = torch.cat([encoder_output_repeat, encoder_output],
                                              dim=0)  # query + targets
            memory_key_padding_mask = torch.cat(
                [torch.ones(n_query_algos_total, encoder_output.shape[1], dtype=torch.bool),
                 torch.zeros(encoder_output.shape[0], encoder_output.shape[1], dtype=torch.bool)],
                dim=0,
            ).to(self.device)
        else:
            encoder_output_repeat = [
                en_o.repeat(n_algo, 1, 1) for en_o, n_algo in zip(encoder_output, n_query_algo)
            ]
            encoder_output_repeat = torch.cat([*encoder_output_repeat, encoder_output],
                                              dim=0)  # query + targets
            memory_key_padding_mask = None

        # assert torch.all(encoder_output_repeat[0] == encoder_output_repeat[1])
        # assert torch.any(encoder_output_repeat[0] != encoder_output_repeat[n_query_algo[0]])

        # [sum(n_algo_test_set) + batch_size, d_model]
        decoder_out = self.lc_decoder(
            tgt=decoder_input,
            memory=encoder_output_repeat,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_padding_masking
        )[:, 0]

        decoder_out_query_algos = torch.split(decoder_out[:-batch_size], n_query_algo.tolist())
        decoder_out_test = decoder_out[-batch_size:]

        # maximal number of algorithm + test algo pads decoder output
        n_algo_max = torch.max(n_query_algo) + 1

        decoder_out_padded = decoder_out_test.new_full(
            (batch_size, n_algo_max, self.d_model),
            fill_value=0)

        # Finally, we only read the output of the last embedding value.
        # Thus, we pad the zero values at the beginning of each sequence.
        for i, decoder_query_algos in enumerate(decoder_out_query_algos):
            decoder_out_padded[i, -n_query_algo[i] - 1: -1] = decoder_out_query_algos[i]
            decoder_out_padded[i, -1] = decoder_out_test[i]

        tgt_mask_decoder_meta = torch.arange(0, n_algo_max)
        tgt_mask_decoder_meta = tgt_mask_decoder_meta < (
                n_algo_max - n_query_algo.unsqueeze(-1) - 1)
        tgt_mask_decoder_meta = tgt_mask_decoder_meta.to(self.device)

        if self.mask_uncorrelated_lcs:
            l_encoder = encoder_output.shape[1]
            l_decoder = decoder_out_padded.shape[1]
            memory_mask = torch.cat(
                [torch.ones(l_decoder - 1, l_encoder, dtype=torch.bool),
                 torch.zeros(1, l_encoder, dtype=torch.bool)],
                dim=0).to(self.device)
        else:
            memory_mask = None

        decoder_output = self.meta_decoder(
            decoder_out_padded,
            memory=encoder_output,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_mask_decoder_meta
        )[:, -1, :]
        prediction = self.readout(decoder_output)

        return prediction

    def select_variable(self, input_variables: List[torch.Tensor], flattened_grn: nn.Module):
        # FIXME: @Difan: documentation?
        combined_feat = torch.stack(input_variables, dim=-1)

        sparse_weights = flattened_grn(torch.cat(input_variables, dim=-1))
        sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)
        meta_embedding = torch.sum(sparse_weights * combined_feat, -1)

        meta_embedding = meta_embedding.view(-1, 1, self.d_model)
        return meta_embedding
