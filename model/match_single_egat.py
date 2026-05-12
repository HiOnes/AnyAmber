import dgl
import torch.nn.functional as F
import torch
import util.utils as utils
from torch import nn
from util.record_func import RecordGraph
from util.graph_constructor import GraphConstructor

from model.match_blocks import (
    DotAttention,
    DirectionEncoder,
    MatchAttentionGNN,
    arange_like,
    log_optimal_transport,
)

class MatchSingleEgat(nn.Module):
    def __init__(self, args, device, return_graph=False, record_graph=True):
        
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

        #### gnn ####
        self.gnn = MatchAttentionGNN(feature_dim=args.others_embed_size, layer_names=['self']*4) # only self edges
        # self.gnn = MatchAttentionGNN(feature_dim=args.others_embed_size, layer_names=['self', 'cross']*4) # self and cross edges
        #### keypoint encoder ####
        self.kenc = DirectionEncoder(args.others_embed_size)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        self.dot_attn = DotAttention(device).to(device)
        self.others_num = args.robot_num - 1
        self.max_cam_num = args.max_cam_num
        self.dim = args.others_embed_size
        self.device = device
        self.max_cov = 10.0
        self.return_graph = return_graph
        self.record_graph = record_graph
        self.RF = None
        if self.record_graph:
            self.RF = RecordGraph(args.wrt_folder, graph_ind=args.wrt_start_g_id, use_uwb_seq=args.record_uwb_seq, add_noise=True) # record graph to csv
        self.GC = None
        if self.return_graph:
            self.GC = GraphConstructor(use_uwb_seq=args.record_uwb_seq) # construct hetero graph

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
        out_match = { 'cam': match_cam, 'cos_similarity': match_cos_similarity, 'var': var, 'indices': indices0, 'scores': scores }
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
    

    def forward(self, batched_graph, batched_msgs):
        dis_seq = batched_graph.ndata['dis_seq']['others'] # [bs*(n+1)*n, dis_len]
        others_feat = torch.cat((batched_graph.ndata['feat']['others'], dis_seq[:, -1].unsqueeze(-1)), dim=-1) # [bs*(n+1)*n, 7+1]

        n, m = self.others_num, self.max_cam_num
        bsn = int(others_feat.shape[0] / n) # bsn = batchsize * (n+1)
        others_prior_pos = others_feat[:, :3] # [bsn*n, 3]
        others_prior_pos = others_prior_pos.reshape(bsn, n, 3) # [bsn, n, 3]
        others_prior_dir = F.normalize(others_prior_pos, p=2.0, dim=-1) # [bsn, n, 3]

        others_cam = batched_graph.ndata['feat']['cam'] # [bsn*m, 3]
        others_cam = others_cam.reshape(bsn, -1, others_cam.shape[1]) # [bsn, m, 3]
        cam_norm2 = torch.norm(others_cam, p=2, dim=2)
        cam_lost_mask = cam_norm2 < 1e-4 # [bsn, m]

        others_encoder = self.kenc(others_prior_dir) # [bsn, dim, n]
        cam_encoder = self.kenc(others_cam) # [bsn, dim, m]
        others_gnn_feat, cam_gnn_feat = self.gnn(others_encoder, cam_encoder) # others_gnn_feat [bsn, dim, n], cam_gnn_feat [bsn, dim, m]
        others_gnn_feat_split, cam_gnn_feat_split = others_gnn_feat.transpose(1,2).reshape(-1,n+1,n,self.dim), cam_gnn_feat.transpose(1,2).reshape(-1,n+1,m,self.dim) # [bs, n+1, n, dim], [bs, n+1, m, dim]
        cam_lost_mask_split = cam_lost_mask.reshape(-1, n+1, m) # [bs, n+1, m]
        others_prior_dir_split = others_prior_dir.reshape(-1, n+1, n, 3) # [bs, n+1, n, 3]
        others_cam_split = others_cam.reshape(-1, n+1, m, 3) # [bs, n+1, m, 3]
        others_feat_split = others_feat.reshape(-1, n+1, n, others_feat.shape[-1]) # [bs, n+1, n, 7+1]
        out_pos = torch.zeros(others_prior_dir_split.shape).to(self.device)
        out_cov = torch.zeros(out_pos.shape[0], n+1, n, 1).to(self.device)
        out_scores = torch.zeros(out_pos.shape[0], n+1, n+1, m+1).to(self.device)
        out_indices = torch.zeros(out_pos.shape[0], n+1, n, 1).to(self.device)
        match_cam = torch.zeros(out_pos.shape[0], n+1, n, 3).to(self.device) # [bs, n+1, n, 3]

        for k in range(n+1):
            others_prior_dir_k = others_prior_dir_split[:, k, :, :] # [bs, n, 3]
            others_cam_k = others_cam_split[:, k, :, :] # [bs, m, 3]
            cam_lost_mask_k = cam_lost_mask_split[:, k, :] # [bs, m]
            others_feat_k = others_feat_split[:, k, :, :] # [bs, n, 7+1]
            dis_lost_mask_k = others_feat_k[:,:,-1] < 1e-4 # [bs, n]
            if (cam_lost_mask_k.all() or dis_lost_mask_k.any()):  # all cams are lost or any distance is lost
                # raise ValueError('all cams are lost or any distance is lost')
                cov_k = torch.ones(others_prior_dir_k.shape[0], n, 1).to(self.device) * self.max_cov
                pos_k = others_feat_k[:,:,:3] # [bs, n, 3]
                scores_k = -torch.inf * torch.ones(others_prior_dir_k.shape[0], n+1, m+1).to(self.device)
                indices_k = -torch.ones(others_prior_dir_k.shape[0], n, 1).to(self.device).to(torch.int64)
                match_cam_k = torch.zeros(others_prior_dir_k.shape[0], n, 3).to(self.device)
            else:
                match_k = self.sinkhorn_match(others_gnn_feat_split[:, k, :, :], cam_gnn_feat_split[:, k, :, :], cam_lost_mask_k, others_cam_k)
                pos_k, cov_k = self.pos_cov_pred(match_k, others_feat_k)
                scores_k, indices_k, match_cam_k = match_k['scores'], match_k['indices'], match_k['cam']

            out_pos[:, k, :, :] = pos_k
            out_cov[:, k, :, :] = cov_k
            out_scores[:, k, :, :] = scores_k
            out_indices[:, k, :, :] = indices_k
            match_cam[:, k, :, :] = match_cam_k
        
        out_pos = out_pos.flatten(0, 1) # [bsn, n, 3]
        out_cov = out_cov.flatten(0, 1) # [bsn, n, 1]
        out_scores = out_scores.flatten(0, 1) # [bsn, n+1, m+1]
        out_indices = out_indices.flatten(0, 1) # [bsn, n, 1]
        match_cam = match_cam.flatten(0, 1) # [bsn, n, 3]

        # if not valid, modify cov and pos
        valid = out_indices > -1
        invalid_cov = torch.tensor(self.max_cov, dtype=out_cov.dtype, device=out_cov.device)
        out_cov = torch.where(valid, out_cov, invalid_cov)
        out_pos = torch.where(valid, out_pos, others_prior_pos)

        outputs = {'pos': out_pos, 'cov': out_cov, 'scores': out_scores, 'indices': out_indices}
        if not (self.record_graph or self.return_graph):
            return outputs

        # bg = self.make_batch_graph(out_pos.reshape(-1, n+1, n, 3), out_cov.reshape(-1, n+1, n, 1), others_feat[:,-1].reshape(-1, n+1, n, 1)) # [bs, n+1, n, 3], [bs, n+1, n, 1], [bs, n+1, n, 1]
        # egat_pos = self.egat(bg) # [bsn, n, 3]

        ### prepare graph ###
        t_decimal = float('0.' + str(batched_msgs['timestamp'][0,1].item())[1:])
        ts = batched_msgs['timestamp'][0,0].item() + t_decimal
        local2map = batched_msgs['local2map'].reshape(-1, n+1, 7) # [bs, n+1, 7]
        world_pose_delta = batched_msgs['world_pose_delta'].reshape(-1, n+1, 7) # [bs, n+1, 7]
        out_pose = torch.cat((out_pos.reshape(-1, n+1, n, 3), others_feat.reshape(-1, n+1, n, others_feat.shape[-1])[..., 3:7]), dim=-1) # [bs, n+1, n, 7]
        label_pose = batched_graph.ndata['label_pos']['others'].reshape(-1, n+1, n, 7) # [bs, n+1, n, 7]
        raw_dis_seq = dis_seq.reshape(-1, n+1, n, dis_seq.shape[-1]) # # [bs, n+1, n, dis_len]
        gt_dis = batched_graph.ndata['gtdis']['others'].reshape(-1, n+1, n, 1) # [bs, n+1, n, 1]
        match_cam = match_cam.reshape(-1, n+1, n, 3) # [bs, n+1, n, 3]
        bearing_mask = (out_indices > -1).reshape(-1, n+1, n) # [bs, n+1, n]
        match_cam[~bearing_mask, :] = 0.0
        ##### label match_cam & bearing_mask & dot_pose #####
        index_gt = batched_graph.ndata['label_match']['others'].reshape(-1, n+1, n) # [bs, n+1, n]
        bearing_mask_gt = index_gt > -1 # [bs, n+1, n]
        label_index = index_gt.unsqueeze(-1).repeat(1, 1, 1, 3) # [bs, n+1, n, 3]
        label_valid = label_index > -1 # [bs, n+1, n, 3]
        invalid_index = torch.tensor(0, dtype=label_index.dtype, device=label_index.device)
        label_index_replace = torch.where(label_valid, label_index, invalid_index) # [bs, n+1, n, 3]
        match_cam_gt = torch.gather(others_cam_split, dim=2, index=label_index_replace) # [bs, n+1, n, 3]
        prior_norm = torch.norm(others_feat_split[..., :3], p=2.0, dim=-1).unsqueeze(-1).repeat(1,1,1,3) # [bs, n+1, n, 3]
        prior_bearing = others_feat_split[..., :3] * (1/prior_norm) # [bs, n+1, n, 3]
        cos_simi = torch.sum(match_cam_gt * prior_bearing, dim=-1, keepdim=True) # [bs, n+1, n, 1]
        simi_valid = (cos_simi > 0.95).repeat(1,1,1,3) # [bs, n+1, n, 3]
        cam_valid = label_valid & simi_valid
        match_cam_gt[~cam_valid] = 0.0
        dis_last = raw_dis_seq[..., -1].unsqueeze(-1).repeat(1, 1, 1, 3) # [bs, n+1, n, 3]
        assert (dis_last > 1e-4).all()
        dot_pos = match_cam_gt * dis_last # [bs, n+1, n, 3]
        dot_pos = torch.where(cam_valid, dot_pos, others_feat_split[..., :3]) # [bs, n+1, n, 3]
        dot_pose = torch.cat((dot_pos, others_feat_split[..., 3:7]), dim=-1) # [bs, n+1, n, 7]
        ##### record graph to csv #####
        # self.RF.record_graph_tolist_ref(dot_pose, label_pose, raw_dis_seq, gt_dis, match_cam_gt, bearing_mask_gt, local2map, world_pose_delta, ts)
        if self.record_graph:
            self.RF.record_graph_tolist_ref(out_pose, label_pose, raw_dis_seq, gt_dis, match_cam, bearing_mask, local2map, world_pose_delta, ts)
        ##### construct hetero graph #####
        # bg = self.GC.construct_graph_ref(dot_pose, label_pose, raw_dis_seq, gt_dis, match_cam_gt, bearing_mask_gt, local2map, world_pose_delta, ts)
        bg = None
        if self.return_graph:
            bg = self.GC.construct_graph_ref(out_pose, label_pose, raw_dis_seq, gt_dis, match_cam, bearing_mask, local2map, world_pose_delta, ts)

        if self.return_graph:
            return outputs, bg
        return outputs

    def make_batch_graph(self, pos, cov, uwb_range):
        '''
        input:
            pos: [bs, n+1, n, 3]
            cov: [bs, n+1, n, 1]
            uwb_range: [bs, n+1, n, 1]
        '''
        bs, n = pos.shape[0], pos.shape[2]
        graph_list = []
        for bi in range(bs):
            for k in range(n+1):
                src_nodes, dst_nodes, e_feats = [], [], []
                # node 0 is [id=k] robot, node 1 ~ n+1 are other robots
                for i in range(n+1):
                    for j in range(n+1):
                        if i == j:
                            continue
                        src_nodes.append(i)
                        dst_nodes.append(j)
                        if i == 0: # k is ref
                            r = uwb_range[bi, k, j-1, 0]
                        elif j == 0: # k is being observed
                            ref_id = utils.ind2id(k, i-1)
                            r = uwb_range[bi, ref_id, utils.id2ind(ref_id, k), 0]
                        else:
                            ref_id = utils.ind2id(k, i-1)
                            this_id = utils.ind2id(k, j-1)
                            r = uwb_range[bi, ref_id, utils.id2ind(ref_id, this_id), 0]
                        e_feats.append(r)
                src_nodes = torch.tensor(src_nodes, dtype=torch.int64).to(self.device)
                dst_nodes = torch.tensor(dst_nodes, dtype=torch.int64).to(self.device)
                e_feats = torch.tensor(e_feats, dtype=torch.float32).to(self.device)
                g = dgl.graph((src_nodes, dst_nodes), num_nodes=n+1)
                g.edata['feat'] = e_feats.unsqueeze(1)
                zero_pos = torch.zeros((1, 3), dtype=pos.dtype).to(self.device)
                p = torch.cat((zero_pos, pos[bi, k]), dim=0) # [1, 3] + [n, 3] = [n+1, 3]
                g.ndata['feat'] = p
                graph_list.append(g)
        batch_graph = dgl.batch(graph_list)
        return batch_graph
