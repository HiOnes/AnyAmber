import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.match_blocks import DotAttention

class MatchHandcrafted(nn.Module):
    def __init__(self, args, device):
        
        super().__init__()        

        self.dot_attn = DotAttention(device).to(device)
        self.others_num = args.robot_num - 1
        self.max_cam_num = args.max_cam_num
        self.dim = args.others_embed_size
        self.device = device
        self.max_cov = 10.0

    def _run_handcrafted_match(self, batched_graph):
        others_feat = batched_graph.ndata['feat']['others'] # [bs*(n+1)*n, 7+1]
        
        n, m = self.others_num, self.max_cam_num
        bsn = int(others_feat.shape[0] / n) # bsn = batchsize * (n+1)
        others_prior_pos = others_feat[:, :3] # [bsn*n, 3]
        others_prior_pos = others_prior_pos.reshape(bsn, n, 3) # [bsn, n, 3]
        others_prior_dir = F.normalize(others_prior_pos, p=2.0, dim=-1) # [bsn, n, 3]

        others_cam = batched_graph.ndata['feat']['cam'] # [bs*(n+1)*n, 3]
        others_cam = others_cam.reshape(bsn, m, others_cam.shape[1]) # [bsn, m, 3]
        cam_norm2 = torch.norm(others_cam, p=2, dim=2)
        cam_lost_mask = cam_norm2 < 1e-4 # [bsn, m]

        cam_lost_mask_split = cam_lost_mask.reshape(-1, n+1, m) # [bs, n+1, m]
        others_prior_dir_split = others_prior_dir.reshape(-1, n+1, n, 3) # [bs, n+1, n, 3]
        others_cam_split = others_cam.reshape(-1, n+1, m, 3) # [bs, n+1, m, 3]
        others_feat_split = others_feat.reshape(-1, n+1, n, others_feat.shape[-1]) # [bs, n+1, n, 7+1]
        out_pos = torch.zeros(others_prior_dir_split.shape).to(self.device)
        out_cov = torch.zeros(out_pos.shape[0], n+1, n, 1).to(self.device)
        out_prob = torch.zeros(out_pos.shape[0], n+1, n, m).to(self.device)
        out_indices = torch.zeros(out_pos.shape[0], n+1, n, 1).to(self.device)

        for k in range(n+1):
            others_prior_dir_k = others_prior_dir_split[:, k, :, :] # [bs, n, 3]
            others_cam_k = others_cam_split[:, k, :, :] # [bs, m, 3]
            cam_lost_mask_k = cam_lost_mask_split[:, k, :] # [bs, m]
            others_feat_k = others_feat_split[:, k, :, :] # [bs, n, 7+1]
            dis_k = others_feat_k[:,:,-1] # [bs, n]
            dis_lost_mask_k = dis_k < 1e-4 # [bs, n]
            if (cam_lost_mask_k.all() or dis_lost_mask_k.any()):  # all cams are lost or any distance is lost
                prob_k = torch.zeros(others_prior_dir_k.shape[0], n, m).to(self.device)
                cov_k = torch.ones(others_prior_dir_k.shape[0], n, 1).to(self.device) * self.max_cov
                pos_k = others_feat_k[:,:,:3] # [bs, n, 3]
                indices_k = -torch.ones(others_prior_dir_k.shape[0], n, 1).to(self.device).to(torch.int64)
            else:
                match_k = self.cos_match(others_prior_dir_k, others_cam_k, cam_lost_mask_k)
                pos_k = dis_k.unsqueeze(2) * match_k['cam'] # [bs, n, 3]
                prob_k, cov_k, indices_k = match_k['prob'], match_k['cov'], match_k['indices']

            out_pos[:, k, :, :] = pos_k
            out_cov[:, k, :, :] = cov_k
            out_prob[:, k, :, :] = prob_k
            out_indices[:, k, :, :] = indices_k
        
        out_pos = out_pos.flatten(0, 1) # [bsn, n, 3]
        out_cov = out_cov.flatten(0, 1) # [bsn, n, 1]
        out_prob = out_prob.flatten(0, 1) # [bsn, n, m]
        out_indices = out_indices.flatten(0, 1) # [bsn, n, 1]

        # if not valid, modify cov and pos
        valid = out_indices > -1
        invalid_cov = torch.tensor(self.max_cov, dtype=out_cov.dtype, device=out_cov.device)
        out_cov = torch.where(valid, out_cov, invalid_cov)
        out_pos = torch.where(valid, out_pos, others_prior_pos)
        out_scores = self._build_scores(out_prob, out_indices)

        outputs = {'prob': out_prob, 'pos': out_pos, 'cov': out_cov, 'scores': out_scores, 'indices': out_indices}

        return outputs

    def _build_scores(self, prob, indices):
        valid = indices > -1
        eps = torch.full_like(indices, 1e-8, dtype=prob.dtype)
        prob = torch.where(valid, prob.clamp_min(1e-8), eps)
        no_match_prob = torch.where(valid, eps, torch.ones_like(indices, dtype=prob.dtype))
        match_scores = torch.log(torch.cat((prob, no_match_prob), dim=-1))
        pad_row = torch.full(
            (match_scores.shape[0], 1, match_scores.shape[2]),
            float("-inf"),
            dtype=match_scores.dtype,
            device=match_scores.device,
        )
        return torch.cat((match_scores, pad_row), dim=1)

    def cos_match(self, prior_dir: Tensor, cam_dir: Tensor, cam_lost_mask: Tensor):
        '''
        input:
            prior_dir: torch.Tensor, [bsn, n, 3]
            cam_dir: torch.Tensor, [bsn, m, 3]
            cam_lost_mask: torch.Tensor, [bsn, m]
        output:
            dict: { 'cam': match_cam [bsn, n, 3], 
                    'prob': prob [bsn, n, m], 
                    'cos_similarity': match_cos_similarity [bsn, n, 1], 
                    'var': var [bsn, n, 1], 
                    'indices': indices [bsn, n, 1], 
                    'cov': cov [bsn, n, 1] }
        '''
        n, m = prior_dir.shape[1], cam_dir.shape[1]
        cos_similarity, prob = self.dot_attn(prior_dir, cam_dir, key_padding_mask=cam_lost_mask) # prob:[bsn, n, m]
        indices = torch.argmax(prob, dim=-1, keepdim=True) # [bsn, n, 1]
        match_cam_index = indices.repeat(1, 1, 3) # [bsn, n, 3]
        match_cam = torch.gather(cam_dir, dim=1, index=match_cam_index) # [bsn, n, 3]
        match_cos_similarity = torch.gather(cos_similarity, dim=2, index=indices) # [bsn, n, 1]
        cov = (1 - match_cos_similarity) * 100.0 # [bsn, n, 1]
        cov = torch.clamp(cov, 0.01, self.max_cov)
        match_valid = match_cos_similarity > 0.99 # [bsn, n, 1]

        attn_dir = torch.bmm(prob, cam_dir) # [bsn, n, 3]
        attn_dir = F.normalize(attn_dir, p=2.0, dim=2) # [bsn, n, 3]
        var = torch.zeros(attn_dir.shape[0], n, 1).to(self.device) # [bsn, n, 1]
        for i in range(n):
            mean = attn_dir[:, i, :].unsqueeze(1) # [bsn, 1, 3]
            gap_mean = cam_dir - mean # [bsn, m, 3]
            gap_square_mean = torch.norm(gap_mean, p=2.0, dim=2, keepdim=True) # [bsn, m, 1]
            bmm = torch.bmm(prob[:,i,:].unsqueeze(1), gap_square_mean) # [bsn, 1, 1]
            var[:, i, :] = bmm.squeeze(1) 
        
        invalid_index = torch.tensor(-1, dtype=indices.dtype, device=indices.device)
        invalid_cov = torch.tensor(self.max_cov, dtype=cov.dtype, device=cov.device)
        indices = torch.where(match_valid, indices, invalid_index)
        cov = torch.where(match_valid, cov, invalid_cov)
        out_match = { 'cam': match_cam, 'prob': prob, 'cos_similarity': match_cos_similarity, 'var': var, 'indices': indices, 'cov': cov }
        return out_match
    

    def forward(self, batched_graph, batched_msgs):
        outputs = self._run_handcrafted_match(batched_graph)
        return outputs
