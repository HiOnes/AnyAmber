import torch.nn.functional as F
import torch
from torch import nn

from model.match_blocks import (
    DotAttention,
    DirectionEncoder,
    FeatureDecoder,
    MatchAttentionGNN,
    arange_like,
    log_optimal_transport,
)

class MatchSplit(nn.Module):
    def __init__(self, args, device):
        
        super().__init__()        
        self.pos_pred = nn.Sequential(
            nn.Linear(3*2+2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.cov_pred = nn.Sequential(
            nn.Linear(3*2+2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self._reset_parameters()

        # gnn
        self.gnn = MatchAttentionGNN(feature_dim=args.others_embed_size, layer_names=['self', 'cross', 'seq']*4, time_win=args.frame_win)
        # keypoint encoder & decoder
        self.kenc = DirectionEncoder(args.others_embed_size)
        self.kdec = FeatureDecoder(args.others_embed_size)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        self.dot_attn = DotAttention(device).to(device)
        self.others_num = args.robot_num - 1
        self.max_cam_num = args.max_cam_num
        self.dim = args.others_embed_size
        self.device = device
        self.max_cov = 10.0

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def sinkhorn_match(self, prior_embed, cam_embed, cam_lost_mask, cam_dir):
        # prior_embed [bs, n, dim] cam_embed [bs, m, dim] cam_lost_mask [bs, m] cam_dir [bs, m, 3]
        n, m = prior_embed.shape[1], cam_embed.shape[1]
        scores, prob = self.dot_attn(prior_embed, cam_embed, key_padding_mask=cam_lost_mask) # [bs, n, m]
        scores = scores / prior_embed.shape[-1] ** .5

         # Run the optimal transport.
        scores = log_optimal_transport(scores, self.bin_score, iters=100) # [bs, n+1, m+1] note!!!

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices # [bs, n], [bs, m]
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0) # [bs, n]
        zero_tensor = torch.tensor(0, dtype=scores.dtype, device=scores.device)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero_tensor) # [bs, n]
        valid0 = mutual0 & (mscores0 > 0.6) # [bsn, n]

        match_cam_index = indices0.unsqueeze(-1).repeat(1, 1, 3) # [bs, n, 3]
        match_cam = torch.gather(cam_dir, dim=1, index=match_cam_index) # [bs, n, 3]
        match_cos_similarity = mscores0.unsqueeze(-1) # [bs, n, 1]

        attn_dir = torch.bmm(prob, cam_dir) # [bs, n, 3]
        attn_dir = F.normalize(attn_dir, p=2.0, dim=2) # [bs, n, 3]
        var = torch.zeros(attn_dir.shape[0], n, 1).to(self.device) # [bs, n, 1]
        for i in range(n):
            mean = attn_dir[:, i, :].unsqueeze(1) # [bs, 1, 3]
            gap = cam_dir - mean # [bs, m, 3]
            gap_square = torch.norm(gap, p=2, dim=2, keepdim=True) # [bs, m, 1]
            bmm = torch.bmm(prob[:,i,:].unsqueeze(1), gap_square) # [bs, 1, 1]
            var[:, i, :] = bmm.squeeze(1) 

        invalid_index = torch.tensor(-1, dtype=indices0.dtype, device=indices0.device)
        indices0 = torch.where(valid0, indices0, invalid_index).unsqueeze(-1) # [bsn, n, 1]
        out_match = { 'cam': match_cam, 'prob': prob, 'cos_similarity': match_cos_similarity, 'var': var, 'indices': indices0, 'scores': scores }
        return out_match
        
    def pos_cov_pred(self, match, others_feat):
        others_d = others_feat[:,:,-1].unsqueeze(2) # [bs, n, 1]
        others_prior_pos = others_feat[:,:,:3] # [bs, n, 3]
        attn_pos = others_d * match['cam'] # [bsn, n, 3]
        pos_feat = torch.cat([others_prior_pos, attn_pos, match['var'], match['cos_similarity']], dim=2) # [bs, n, 3*2+2]
        cov = self.cov_pred(pos_feat) # [bs, n, 1]
        cov = torch.clamp(cov, 1e-4, self.max_cov) # [bs, n, 1] note!!!
        outputs_relative_pos = self.pos_pred(pos_feat) # [bs, n, 3]
        outputs_pos = others_prior_pos + outputs_relative_pos # [bs, n, 3]
        return outputs_pos, cov
    
    def forward(self, graph_seq):
        # graph_seq is list of length [time_window], each element is graph, each graph feat is size [batch, feat]
        time_win = len(graph_seq) # tw
        n, m, bsn = self.others_num, self.max_cam_num, 0
        others_feat, others_cam = None, None # [tw*bs*(n+1)*n, 7+1], [tw*bs*(n+1)*m, 3]
        for g in range(time_win):
            if others_feat is None:
                others_feat = graph_seq[g].ndata['feat']['others']
                others_cam = graph_seq[g].ndata['feat']['cam']
                bsn = int(others_feat.shape[0] / n) # bsn = batchsize * (n+1)
            else:
                others_feat = torch.cat((others_feat, graph_seq[g].ndata['feat']['others']), dim=0)
                others_cam = torch.cat((others_cam, graph_seq[g].ndata['feat']['cam']), dim=0)
        
        others_prior_pos = others_feat[:, :3] # [tw*bsn*n, 3]
        others_prior_pos = others_prior_pos.reshape(time_win*bsn, n, 3) # [tw*bsn, n, 3] 
        others_prior_dir = F.normalize(others_prior_pos, p=2.0, dim=-1) # [tw*bsn, n, 3]
        others_cam = others_cam.reshape(time_win*bsn, m, others_cam.shape[-1]) # [tw*bsn, m, 3]
        cam_norm2 = torch.norm(others_cam, p=2, dim=-1)
        cam_lost_mask = cam_norm2 < 1e-4 # [tw*bsn, m]

        ########### gnn ############
        others_encoder, cam_encoder = self.kenc(others_prior_dir), self.kenc(others_cam) # [tw*bsn, dim, n], [tw*bsn, dim, m]
        others_gnn_feat, cam_gnn_feat = self.gnn(others_encoder, cam_encoder) # others_gnn_feat [tw*bsn, dim, n], cam_gnn_feat [tw*bsn, dim, m]
        others_gnn_feat, cam_gnn_feat = self.kdec(others_gnn_feat), self.kdec(cam_gnn_feat) # [tw*bsn, dim, n], [tw*bsn, dim, m]
        others_gnn_feat_split, cam_gnn_feat_split = others_gnn_feat.transpose(1,2).reshape(-1,n+1,n,self.dim), cam_gnn_feat.transpose(1,2).reshape(-1,n+1,m,self.dim) # [tw*bs, n+1, n, dim], [tw*bs, n+1, m, dim]
        cam_lost_mask_split = cam_lost_mask.reshape(-1, n+1, m) # [tw*bs, n+1, m]
        others_prior_dir_split = others_prior_dir.reshape(-1, n+1, n, 3) # [tw*bs, n+1, n, 3]
        others_cam_split = others_cam.reshape(-1, n+1, m, 3) # [tw*bs, n+1, m, 3]
        others_feat_split = others_feat.reshape(-1, n+1, n, others_feat.shape[-1]) # [tw*bs, n+1, n, 7+1]
        out_pos = torch.zeros(others_prior_dir_split.shape).to(self.device) # [tw*bs, n+1, n, 3]
        out_cov = torch.zeros(out_pos.shape[0], n+1, n, 1).to(self.device) # [tw*bs, n+1, n, 1]
        out_prob = torch.zeros(out_pos.shape[0], n+1, n, m).to(self.device) # [tw*bs, n+1, n, m]
        out_scores = torch.zeros(out_pos.shape[0], n+1, n+1, m+1).to(self.device) # [tw*bs, n+1, n+1, m+1]
        out_indices = torch.zeros(out_pos.shape[0], n+1, n, 1).to(self.device) # [tw*bs, n+1, n, 1]

        for k in range(n+1):
            others_prior_dir_k = others_prior_dir_split[:, k, :, :] # [tw*bs, n, 3]
            others_cam_k = others_cam_split[:, k, :, :] # [tw*bs, m, 3]
            cam_lost_mask_k = cam_lost_mask_split[:, k, :] # [tw*bs, m]
            others_feat_k = others_feat_split[:, k, :, :] # [tw*bs, n, 7+1]
            dis_lost_mask_k = others_feat_k[:,:,-1] < 1e-4 # [tw*bs, n]
            if (cam_lost_mask_k.all() or dis_lost_mask_k.any()):  # all cams are lost or any distance is lost
                prob_k = torch.zeros(others_prior_dir_k.shape[0], n, m).to(self.device)
                cov_k = torch.ones(others_prior_dir_k.shape[0], n, 1).to(self.device) * self.max_cov
                pos_k = others_feat_k[:,:,:3] 
                scores_k = -torch.inf * torch.ones(others_prior_dir_k.shape[0], n+1, m+1).to(self.device)
                indices_k = -torch.ones(others_prior_dir_k.shape[0], n, 1).to(self.device).to(torch.int64)
            else:
                match_k = self.sinkhorn_match(others_gnn_feat_split[:, k, :, :], cam_gnn_feat_split[:, k, :, :], cam_lost_mask_k, others_cam_k)
                pos_k, cov_k = self.pos_cov_pred(match_k, others_feat_k)
                prob_k, scores_k, indices_k = match_k['prob'], match_k['scores'], match_k['indices']

            out_pos[:, k, :, :] = pos_k
            out_cov[:, k, :, :] = cov_k
            out_prob[:, k, :, :] = prob_k
            out_scores[:, k, :, :] = scores_k
            out_indices[:, k, :, :] = indices_k
        
        out_pos = out_pos.flatten(0, 1) # [tw*bsn, n, 3]
        out_cov = out_cov.flatten(0, 1) # [tw*bsn, n, 1]
        out_prob = out_prob.flatten(0, 1) # [tw*bsn, n, m]
        out_scores = out_scores.flatten(0, 1) # [tw*bsn, n+1, m+1]
        out_indices = out_indices.flatten(0, 1) # [tw*bsn, n, 1]

        # if not valid, modify cov and pos
        valid = out_indices > -1
        invalid_cov = torch.tensor(self.max_cov, dtype=out_cov.dtype, device=out_cov.device)
        out_cov = torch.where(valid, out_cov, invalid_cov)
        out_pos = torch.where(valid, out_pos, others_prior_pos)

        outputs = {'prob': out_prob, 'pos': out_pos, 'cov': out_cov, 'scores': out_scores, 'indices': out_indices}
        return outputs
