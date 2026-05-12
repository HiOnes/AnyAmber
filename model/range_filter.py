import torch
from typing import Optional, Tuple
from copy import deepcopy
from torch import nn

from model.match_blocks import build_mlp

class RangeEncoder(nn.Module):
    """Encode a scalar range sequence into per-step features."""

    def __init__(self, in_dim: int, feature_dim: int):
        super().__init__()
        self.encoder = build_mlp([in_dim, 32, 64, 64] + [feature_dim], do_bn=False)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts: torch.Tensor) -> torch.Tensor:
        return self.encoder(kpts.transpose(1, 2))


class RangeGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.mlp_filter = nn.Sequential(
            nn.Linear(self.hidden_size*1, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )
        self.mlp_cov = deepcopy(self.mlp_filter)

    def forward(
        self,
        input_seq: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        edge_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict filtered range and covariance from a distance sequence.

        Args:
            input_seq (torch.Tensor): [bs, dis_len, dim]
            h (torch.Tensor, optional): [num_layers, bs, hidden_size]
            edge_embed (torch.Tensor, optional): [bs, hidden_size]

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                filter_range [bs, 1], range_cov [bs, 1], h_n [num_layers, bs, hidden_size]
        """
        output, h_n = self.gru(input_seq, h)
        output = output[:, -1, :]
        if edge_embed is not None:
            output = output + edge_embed
        filter_range = self.mlp_filter(output)
        range_cov = self.mlp_cov(output)
        return filter_range, range_cov, h_n

class RangeFilter(nn.Module):
    def __init__(self, max_num, hidden_size, device):
        super().__init__()
        self.embedding = nn.Embedding(max_num, hidden_size)
        self.range_enc = RangeEncoder(1, hidden_size)
        self.gru_range = RangeGRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=3).to(device)
        self.device = device
        self.max_cov = 10.0
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        raw_range: torch.Tensor,
        edge_id: Optional[torch.Tensor] = None,
        use_sensor_embedding: bool = False,
        pair_embedding: Optional[bool] = None,
    ) -> dict:
        """Run range filtering with optional sensor-pair embedding.

        Args:
            raw_range (torch.Tensor): [bs, dis_seq_len]
            edge_id (torch.Tensor, optional): [bs, 2] sensor pair indices.
            use_sensor_embedding (bool): Whether to add pair embedding.

        Returns:
            dict: {"filter_range": [bs, 1], "range_cov": [bs, 1], "h": [layers, bs, dim]}
        """
        if pair_embedding is not None:
            use_sensor_embedding = pair_embedding

        edge_embed = None
        if use_sensor_embedding:
            assert edge_id is not None, "edge_id must be provided for pair embedding."
            src_embed = self.embedding(edge_id[:, 0])
            dst_embed = self.embedding(edge_id[:, 1])
            edge_embed = src_embed + dst_embed

        dis_seq = raw_range.unsqueeze(-1) # [bs, dis_seq_len, 1]
        others_range_feat = self.range_enc(dis_seq) # [bs, dim, dis_seq_len]
        others_range_feat = others_range_feat.transpose(1,2) # [bs, dis_seq_len, dim]
        filter_range, range_cov, h = self.gru_range(others_range_feat, edge_embed=edge_embed) # [bs, 1]
        range_cov = torch.clamp(range_cov, 1e-4, self.max_cov)

        return {"filter_range": filter_range, "range_cov": range_cov, "h": h}
