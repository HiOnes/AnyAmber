import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch
from copy import deepcopy

class Egat(nn.Module):
    def __init__(self, node_in_dict, edge_in_dict, hid_dim, gat_heads, gat_layers, need_dis_cov=False, aggr_fn='mean'):
        """
        Heterogeneous EdgeGAT model for regression tasks.
        Args:
            node_in_dict: dict, Dictionary mapping node types to their input feature sizes.
            edge_in_dict: dict, Dictionary mapping edge types to their input feature sizes.
            hid_dim, int, Hidden dimension size for the GAT layers.
            gat_heads, int, Number of attention heads in the GAT layers.
            gat_layers, int, Number of GAT layers to stack.
            need_dis_cov, bool, retained for compatibility. This base Egat
                class does not emit dis_cov; use EgatLoading for that path.
            aggr_fn, str, Aggregation function for the GAT layers, allows 'mean' or 'stack'.
        """

        super().__init__()
        assert aggr_fn in ['mean', 'stack'], "Aggregation function must be either 'mean' or 'stack'."

        ##### projection layers #####
        self.n_proj = nn.ModuleDict({
            ntype: nn.Linear(in_size, hid_dim)
            for ntype, in_size in node_in_dict.items()
        })

        ##### Hetero EdgeGAT layers #####
        self.hetero_module = dglnn.HeteroGraphConv({
            rel: dglnn.EdgeGATConv((hid_dim, hid_dim), edge_in_dict[rel], hid_dim, gat_heads)
            for rel in edge_in_dict.keys()}, aggregate=aggr_fn)
        self.egat_layers = nn.ModuleList([deepcopy(self.hetero_module) for _ in range(gat_layers)])

        ##### merge layers #####
        n_in_edgetype_dict = { k: 0 for k in node_in_dict.keys() }
        for etype in edge_in_dict.keys():
            n_in_edgetype = etype.split('2')[-1]  # e.g., 'moving2fixed' -> 'fixed'
            n_in_edgetype_dict[n_in_edgetype] += 1
        if aggr_fn == 'mean':
            merge_unit = nn.ModuleDict({
                ntype: nn.Sequential(
                    nn.Linear(hid_dim*gat_heads, hid_dim),
                    nn.ReLU(),
                    nn.Linear(hid_dim, hid_dim),
                    nn.ReLU()
                ) for ntype, in_edge in n_in_edgetype_dict.items() if in_edge > 0})
        elif aggr_fn == 'stack':
            merge_unit = nn.ModuleDict({
                ntype: nn.Sequential(
                    nn.Linear(hid_dim*gat_heads*in_edge, hid_dim),
                    nn.ReLU(),
                    nn.Linear(hid_dim, hid_dim),
                    nn.ReLU()
                ) for ntype, in_edge in n_in_edgetype_dict.items() if in_edge > 0})
        self.merge_layers = nn.ModuleList([deepcopy(merge_unit) for _ in range(gat_layers)])

        ##### MLP layers for regression outputs #####
        self.mlp_moving_pos = nn.Sequential(
            nn.Linear(hid_dim, int(hid_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hid_dim/2), 3)
        )
        self.mlp_moving_cov = nn.Sequential(
            nn.Linear(hid_dim, int(hid_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hid_dim/2), 3),
            nn.Sigmoid()
        )
        self.mlp_base_pos = nn.Sequential(
            nn.Linear(hid_dim, int(hid_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hid_dim/2), 3)
        )
        self.mlp_base_quat = nn.Sequential(
            nn.Linear(hid_dim, int(hid_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hid_dim/2), 4),
        )
        self.mlp_base_cov = nn.Sequential(
            nn.Linear(hid_dim, int(hid_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hid_dim/2), 6),
            nn.Sigmoid()
        )
        self.mlp_dis_cov = nn.Sequential(
            nn.Linear(hid_dim*2+1, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, int(hid_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hid_dim/2), 1),
            nn.Sigmoid()
        )
        self.max_cov = 10.0
        self.need_dis_cov = need_dis_cov

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, graph, nodes_retain_dict=None, initial=False):
        if isinstance(nodes_retain_dict, bool):
            initial = nodes_retain_dict
            nodes_retain_dict = None
        g = graph.clone()
        moving_pos_list, base_pose_list = [], []
        moving_mask, base_mask = g.ndata['valid']['moving'].bool(), g.ndata['valid']['base'].bool()

        if initial:
            ### Initial state, no GAT layers applied ###
            moving_pos = g.ndata['feat']['moving'][moving_mask, :3]
            moving_pos_list.append(moving_pos)
            moving_cov = torch.ones_like(moving_pos)
            base_pose = g.ndata['feat']['base'][base_mask, :7]
            base_pose_list.append(base_pose)
            base_cov = torch.ones((base_pose.shape[0], 6), device=g.device, dtype=base_pose.dtype)
        else:
            ### Project node features to hidden dimension ###
            g.ndata['feat'] = { k: self.n_proj[k](v) for k, v in g.ndata['feat'].items() }
            ### Apply GAT layers and merge outputs ###
            for egat_layer, merge_layer in zip(self.egat_layers, self.merge_layers):
                e_feat_dict = {'fixed2moving': (g.edata['feat'][('fixed','fixed2moving','moving')],), 
                            'moving2fixed': (g.edata['feat'][('moving','moving2fixed','fixed')],),
                            'moving2moving': (g.edata['feat'][('moving','moving2moving','moving')],)}
                e_feat_dict['moving2base'] = (g.edata['feat'][('moving','moving2base','base')],)
                e_feat_dict['ref2base'] = (g.edata['feat'][('ref','ref2base','base')],)
                n_new = egat_layer(g, g.ndata['feat'], mod_args=e_feat_dict)
                g.ndata['feat'] = {k: merge_layer[k](v.reshape(v.shape[0],-1)) for k, v in n_new.items()}

            moving_pos = self.mlp_moving_pos(g.ndata['feat']['moving'])[moving_mask]
            moving_pos_list.append(moving_pos)
            moving_cov = torch.clamp(self.mlp_moving_cov(g.ndata['feat']['moving'])[moving_mask], 1e-4, self.max_cov)
            base_pos = self.mlp_base_pos(g.ndata['feat']['base'])[base_mask]
            base_quat = self.mlp_base_quat(g.ndata['feat']['base'])[base_mask]
            base_pose = torch.cat((base_pos, base_quat), dim=-1)
            base_cov = torch.clamp(self.mlp_base_cov(g.ndata['feat']['base'])[base_mask], 1e-4, self.max_cov)
            base_pose_list.append(base_pose)

        return {'moving_pos': moving_pos_list, 'moving_cov': moving_cov, 'base_pose': base_pose_list, 'base_cov': base_cov}
