from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def build_mlp(channels: list, do_bn: bool = True) -> nn.Module:
    layers = []
    for i in range(1, len(channels)):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < len(channels) - 1:
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim**0.5
    prob = F.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        batch_dim = query.size(0)
        query, key, value = [
            layer(x).view(batch_dim, self.dim, self.num_heads, -1)
            for layer, x in zip(self.proj, (query, key, value))
        ]
        x, _ = _scaled_dot_product_attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, feature_dim)
        self.mlp = build_mlp([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: Tensor, source: Tensor) -> Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class MatchAttentionGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, time_win: Optional[int] = None):
        super().__init__()
        self.layers = nn.ModuleList(
            [AttentionPropagation(feature_dim, 4) for _ in range(len(layer_names))]
        )
        self.names = layer_names
        self.tw = time_win

    def forward(self, desc0: Tensor, desc1: Tensor) -> Tuple[Tensor, Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == "self":
                delta0, delta1 = layer(desc0, desc0), layer(desc1, desc1)
                desc0, desc1 = desc0 + delta0, desc1 + delta1
            elif name == "cross":
                delta0, delta1 = layer(desc0, desc1), layer(desc1, desc0)
                desc0, desc1 = desc0 + delta0, desc1 + delta1
            elif name == "seq":
                if self.tw is None:
                    raise ValueError("time_win must be set when using seq attention layers.")
                d0 = desc0.reshape(self.tw, -1, desc0.shape[-2], desc0.shape[-1])
                d1 = desc1.reshape(self.tw, -1, desc1.shape[-2], desc1.shape[-1])
                d0_update, d1_update = d0.clone(), d1.clone()
                for i in range(self.tw):
                    for j in range(self.tw):
                        if i == j:
                            continue
                        delta0, delta1 = layer(d0[i], d0[j]), layer(d1[i], d1[j])
                        d0_update[i], d1_update[i] = d0_update[i] + delta0, d1_update[i] + delta1
                desc0, desc1 = d0_update.reshape(desc0.shape), d1_update.reshape(desc1.shape)
            else:
                raise ValueError(f"Unknown layer name: {name}")
        return desc0, desc1


def log_sinkhorn_iterations(Z: Tensor, log_mu: Tensor, log_nu: Tensor, iters: int) -> Tensor:
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: Tensor, alpha: Tensor, iters: int) -> Tensor:
    b, m, n = scores.shape
    one = torch.tensor(1, dtype=scores.dtype, device=scores.device)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat(
        [torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1
    )

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    return Z - norm


def arange_like(x: Tensor, dim: int) -> Tensor:
    return x.new_ones(x.shape[dim]).cumsum(0) - 1


class DirectionEncoder(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.encoder = build_mlp([3, 32, 64, 64, feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts: Tensor) -> Tensor:
        return self.encoder(kpts.transpose(1, 2))


class FeatureDecoder(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.decoder = build_mlp([feature_dim, feature_dim, feature_dim])
        nn.init.constant_(self.decoder[-1].bias, 0.0)

    def forward(self, kpts: Tensor) -> Tensor:
        return self.decoder(kpts)


class DotAttention(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if key_padding_mask is not None:
            mask = key_padding_mask.float()
            mask = mask.masked_fill(key_padding_mask, float("-inf")).unsqueeze(1)
            cos_similarity = torch.baddbmm(mask, query, key.transpose(-2, -1))
        else:
            cos_similarity = torch.bmm(query, key.transpose(-2, -1))
        attn_weight = F.softmax(cos_similarity, dim=-1)
        return cos_similarity, attn_weight
