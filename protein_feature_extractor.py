import math

import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
import torch_geometric
from Bio.PDB.Polypeptide import three_to_index


def _normalize(tensor, dim=-1):
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(x, d_min=0., d_max=20., d_count=16, device='cpu'):
    D_mu = torch.linspace(d_min, d_max, d_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (d_max - d_min) / d_count
    D_expand = torch.unsqueeze(x, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def trainable_rbf(x, t_sigma, d_min, d_max=7.0, d_count=16, device='cpu'):
    D_mu = torch.linspace(d_min, d_max, d_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_expand = torch.unsqueeze(x, -1)

    RBF = torch.exp(-((D_expand - D_mu) / t_sigma) ** 2)
    return RBF


class PDBFeatures(object):
    top_k = 32
    num_positional_embeddings = 16
    num_rbf = 16
    augment_eps = 0.01
    rbf = []

    @staticmethod
    def sequence_to_index(sequence):
        sequence_index = []
        for items in sequence:
            try:
                idx = three_to_index(items)
            except KeyError:
                idx = 20
            assert 0 <= idx <= 20
            sequence_index.append(idx)
        return torch.as_tensor(sequence_index, dtype=torch.long)

    def extract_features(self, features, sequence, is_train=False):
        if type(sequence) != torch.Tensor:
            sequence_idx = self.sequence_to_index(sequence)
        else:
            sequence_idx = sequence
        coords = torch.as_tensor(features, dtype=torch.float32)

        if is_train and self.augment_eps > 0:
            coords = coords + self.augment_eps * torch.randn_like(coords)
        mask = torch.isfinite(coords.sum(dim=(1, 2)))
        coords[~mask] = np.inf
        mask = torch.isfinite(coords.sum(dim=(1, 2)))
        coords[~mask] = np.inf
        ca_coords = coords[:, 1]
        c_coords = coords[:, 2]
        edge_index = torch_cluster.knn_graph(ca_coords, k=self.top_k)
        v_c = self._orientations(c_coords)
        vc_norm = torch.pairwise_distance(v_c[:, 0, :], v_c[:, 1, :])
        vc_norm = vc_norm.unsqueeze(-1)
        cos_vc = self._v_cosine(v_c[:, 0, :], v_c[:, 1, :])
        cos_vc = cos_vc.unsqueeze(-1)
        s_c = torch.cat((vc_norm, cos_vc), dim=-1)
        s_c, v_c = map(torch.nan_to_num, (s_c, v_c))
        pos_embeddings = self._positional_embeddings(edge_index)
        E_vectors = ca_coords[edge_index[0]] - ca_coords[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), d_count=self.num_rbf)
        dihedrals = self._dihedrals(coords)
        orientations = self._orientations(ca_coords)
        sidechains = self._sidechains(coords)

        s_ca = dihedrals
        v_ca = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
        edge_v = _normalize(E_vectors).unsqueeze(-2)
        s_ca, v_ca, edge_s, edge_v = map(torch.nan_to_num,
                                         (s_ca, v_ca, edge_s, edge_v))

        data = torch_geometric.data.Data(x=ca_coords, sequence=sequence_idx,
                                         node_no_s=s_c,
                                         node_no_v=v_c,
                                         node_s=s_ca, node_v=v_ca,
                                         edge_s=edge_s, edge_v=edge_v,
                                         mask=mask,
                                         edge_index=edge_index)

        return data

    @staticmethod
    def _dihedrals(x, eps=1e-7):
        x = torch.reshape(x[:, :3], [3 * x.shape[0], 3])
        dX = x[1:] - x[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def _positional_embeddings(self, edge_index,
                               num_embeddings=None):
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    @staticmethod
    def _orientations(x):
        forward = _normalize(x[1:] - x[:-1])
        backward = _normalize(x[:-1] - x[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    @staticmethod
    def _sidechains(X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)

        return vec

    @staticmethod
    def _v_cosine(a, b):
        inner_product = (a * b).sum(dim=1)
        a_norm = a.pow(2).sum(dim=1).pow(0.5)
        b_norm = b.pow(2).sum(dim=1).pow(0.5)
        cos = inner_product / (2 * a_norm * b_norm)
        return cos


