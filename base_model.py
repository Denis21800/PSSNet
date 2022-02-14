import torch
from attention import SelfAttention, GAT
from gvp_layers import GVP, GVPConvLayer, LayerNorm
import torch_cluster, torch_geometric, torch_scatter
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig

aa_max_len = 21


class PSSModel(nn.Module):

    def __init__(self, shortcut=False):
        super().__init__()
        config = ModelConfig()
        vx_input_ca = config.model_params['vx_input_ca']
        vx_input_c = config.model_params['vx_input_c']
        vx_h_ca = config.model_params['vx_h_ca']
        vx_h_c = config.model_params['vx_h_ca']
        ex_input = config.model_params['ex_input']
        ex_hidden = config.model_params['ex_hidden']
        drop_rate = config.model_params['drop_rate']
        self.use_gat = config.model_params['graph_attention']

        n_layers = config.n_post_processing_layers if shortcut else config.n_base_processing_layers
        self.shortcut = shortcut
        self.embedding = nn.Embedding(aa_max_len, aa_max_len)
        self.gvp_edge = nn.Sequential(
            GVP(ex_input, ex_hidden, activations=(None, None)),
            LayerNorm(ex_hidden)
        )

        self.gvp_v_ca = nn.Sequential(
            GVP(vx_input_ca, vx_h_ca, activations=(None, None)),
            LayerNorm(vx_h_ca)
        )
        self.gvp_v_c = nn.Sequential(
            GVP(vx_input_c, vx_h_c, activations=(None, None)),
            LayerNorm(vx_h_c)
        )
        hvx_dim = (vx_h_ca[0] + vx_h_c[0], vx_h_ca[1] + vx_h_c[1])
        ex_hidden = (ex_hidden[0] + aa_max_len, ex_hidden[1])

        self.encoder = nn.ModuleList(
            GVPConvLayer(hvx_dim, ex_hidden, drop_rate=drop_rate, vector_gate=self.shortcut)
            for _ in range(n_layers))

        gvp_inference_dim = hvx_dim
        rnn_encoder_dim = config.model_params['rnn_encoder_dim']
        rnn_decoder_dim = config.model_params['rnn_decoder_dim']
        if not self.shortcut:
            self.rnn_encoder_norm = nn.LayerNorm(hvx_dim[0])
            self.rnn_decoder_norm = nn.LayerNorm(config.model_params['self_att_dim'])

        self.bi_rnn_encoder = nn.GRU(hvx_dim[0], rnn_encoder_dim, 2, True, True, drop_rate, True)
        self.bi_rnn_decoder = nn.ModuleList(nn.GRU(config.model_params['self_att_dim'], rnn_decoder_dim,
                                                   bidirectional=True, batch_first=True, bias=True)
                                            for _ in range(n_layers))

        self.self_att = SelfAttention(2 * rnn_encoder_dim, config.model_params['self_att_dim'])
        hvx_dim = (config.model_params['self_att_dim'], hvx_dim[1])
        self.decoder_layers = nn.ModuleList(
            GVPConvLayer(hvx_dim, ex_hidden,
                         drop_rate=drop_rate, autoregressive=True)
            for _ in range(n_layers))

        self.gvp_segmentation = nn.Sequential(
            LayerNorm(hvx_dim),
            GVP(hvx_dim, (1, 0),
                activations=(None, None))
        )

        if self.shortcut:
            inference_dim = config.model_params['inference_dim']
            self.graph_att = GAT(in_dim=gvp_inference_dim[0],
                                 out_dim=gvp_inference_dim[0])
            self.gvp_inference = nn.Sequential(
                LayerNorm(gvp_inference_dim),
                GVP(gvp_inference_dim, (inference_dim, 0),
                    activations=(F.leaky_relu, F.leaky_relu))
            )

            self.inference = nn.Sequential(
                nn.Linear(inference_dim * 3, inference_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(inference_dim // 2, 1),
                nn.Sigmoid()
            )

    def forward(self, batch):
        embedding = self.embedding(batch.sequence)
        hx_ca = (batch.node_s, batch.node_v)
        hx_edge = (batch.edge_s, batch.edge_v)
        hx_c = (batch.node_no_s, batch.node_no_v)
        hx_c = self.gvp_v_c(hx_c)
        hx_ca = self.gvp_v_ca(hx_ca)
        hx_edge = self.gvp_edge(hx_edge)
        hV0 = torch.cat((hx_ca[0], hx_c[0]), dim=-1)
        hv1 = torch.cat((hx_ca[1], hx_c[1]), dim=-2)
        hx_vx = (hV0, hv1)
        hx_embedding = embedding[batch.edge_index[0]]
        hx_embedding[batch.edge_index[0] >= batch.edge_index[1]] = 0
        hx_edge = (torch.cat([hx_edge[0], hx_embedding], dim=-1), hx_edge[1])

        for i, layer in enumerate(self.encoder):
            hx_vx = layer(hx_vx, batch.edge_index, hx_edge)
        if self.shortcut:
            return self.__shortcut_fw(batch, hx_vx)

        rnn_input = hx_vx[0].unsqueeze(0)
        rnn_input = self.rnn_encoder_norm(rnn_input)
        rnn_output, _ = self.bi_rnn_encoder(rnn_input)
        rnn_output = self.self_att(rnn_output.transpose(1, 2)).squeeze(0)
        rnn_output = self.rnn_decoder_norm(rnn_output)
        hx_vx = (rnn_output, hx_vx[1])
        encoder_vector = hx_vx

        for i, layer in enumerate(self.decoder_layers):
            hx_vx = layer(hx_vx, batch.edge_index, hx_edge, autoregressive_x=encoder_vector)
            rnn_input = hx_vx[0].unsqueeze(0)
            rnn_output, _ = self.bi_rnn_decoder[i](rnn_input)
            rnn_output = rnn_output.squeeze(0)
            hx_vx = (rnn_output, hx_vx[1])

        out_segmentation = self.gvp_segmentation(hx_vx)
        out_segmentation = torch.sigmoid(out_segmentation)
        return out_segmentation

    def __shortcut_fw(self, batch, hv):
        batch_id = batch.batch
        if self.use_gat:
            hvs = self.gat(hv[0], batch.edge_index)
            hv = (hvs, hv[1])
        out_v = self.gvp_inference(hv)
        out_scatter = torch_scatter.scatter(out_v, batch_id, dim=0)
        out_mean = torch_scatter.scatter_mean(out_v, batch_id, dim=0)
        out_max, _ = torch_scatter.scatter_max(out_v, batch_id, dim=0)
        out_ = torch.cat((out_mean, out_scatter, out_max), dim=1)
        inference = self.inference(out_)
        return inference
