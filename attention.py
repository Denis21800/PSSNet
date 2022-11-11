import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class SelfAttention(nn.Module):
    def __init__(self, in_channel, out_channel=None, attn_dropout=0.1):
        super(SelfAttention, self).__init__()
        self.in_channel = in_channel

        if out_channel is not None:
            self.out_channel = out_channel
        else:
            self.out_channel = in_channel

        self.temperature = self.out_channel ** 0.5

        self.q_map = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.k_map = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.v_map = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        q = self.q_map(x)
        k = self.k_map(x)
        v = self.v_map(x)

        attn = torch.matmul(q.transpose(1, 2) / self.temperature, k)
        attn = self.dropout(F.softmax(attn, dim=-1))
        y = torch.matmul(attn, v.transpose(1, 2))
        return y


class GAT(torch.nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 4
        self.activation = activation

        self.conv1 = GATConv(in_dim, self.hid, heads=self.in_head, dropout=0.3)
        self.conv2 = GATConv(self.hid * self.in_head, out_channels=out_dim, concat=False,
                             heads=self.out_head, dropout=0.3)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        if not self.activation:
            return x

        return self.activation(x)