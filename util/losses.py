import torch.nn.functional as F
from torch import nn
import torch
import util.utils as utils

class SetLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss_weights = {
            'match': 1.0,
            'pos': 1.0,
            'rot': 1.0,
            'tagpos': 1.0,
            'range': 1.0,
            'cov': 0.1,
            'cov_det': 0.04,
        }

    def _combine_total(self, components):
        total = None
        for name, value in components.items():
            if value is None:
                continue
            weighted = value * self.loss_weights[name]
            total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError("No loss components to combine")
        return total
    
    def forward(self, out, target, supervised_type):
        if supervised_type == 'match_6dpose':
            return self.forward_match_6dpose(out, target)
        elif supervised_type == "match_6dpose_compact":
            return self.forward_match_6dpose_compact(out, target)
        elif supervised_type == "match_6dpose_compact_egat":
            return self.forward_match_6dpose_compact_egat(out, target)
        elif supervised_type == "handcrafted":
            return self.forward_handcrafted(out, target)
        elif supervised_type == 'scores_3dpos_cov':
            return self.forward_scores_3dpos_cov(out, target)
        elif supervised_type == 'scores_3dpos_cov_compact':
            return self.forward_scores_3dpos_cov_compact(out, target)
        elif supervised_type == 'scores_3dpos_cov_seq':
            return self.forward_scores_3dpos_cov_seq(out, target)
        elif supervised_type == 'range':
            return self.forward_range(out, target)
        elif supervised_type == 'ref':
            return self.forward_ref(out, target)
        elif supervised_type == 'ref_pgo':
            return self.forward_ref_pgo(out, target)
        else:
            raise ValueError("Invalid supervised_type")
    
    def forward_scores_3dpos_cov(self, out, target):
        '''
        input:
            out: dict
                scores: torch.tensor, [bs*(n+1), n+1, m+1] in log space negative
                indices: torch.tensor, [bs*(n+1), n, 1]
                cov: torch.tensor, [bs*(n+1), n, 1]
                pos: torch.tensor, [bs*(n+1), n, 3]
            target: dict
                pose: torch.tensor, [bs*(n+1)*n, 7]
                match: torch.tensor, [bs*(n+1)*n, 1] others2cam
                match_cam: torch.tensor, [bs*(n+1)*m, 1] cam2others
        output:
            loss: dict
        '''

        n, m = out['scores'].size(1) - 1, out['scores'].size(2) - 1

        ################### match loss ##################
        label_indices = {'src2des': target['match'], 'des2src': target['match_cam']}
        match_others_cam = self.loss_match_func(label_indices, out['scores'], out['indices'])
        loss_match = match_others_cam['cost_src2des'] + match_others_cam['cost_des2src']
        recall, precision = match_others_cam['recall'], match_others_cam['precision']

        ################### pose loss ##################
        label_pose = target['pose'] # [bs*(n+1)*n, 7]
        out_pos = out['pos'].flatten(0,1) # [bs*(n+1)*n, 3]
        out_cov = out['cov'].flatten(0,1) # [bs*(n+1)*n, 1]
        loss_pos = F.mse_loss(out_pos[:, :3], label_pose[:, :3]) * 3
        loss_rot = torch.tensor(0.0).to(loss_pos.device)
        loss_pose = loss_pos

        ################### cov loss ##################
        loss_cov = self.loss_cov_func(label_pose[:, :3], out_pos, out_cov, match_others_cam['pred_valid_mask'], dim=3)

        ################ final loss ################
        total_loss = self._combine_total({'match': loss_match, 'pos': loss_pose, 'cov': loss_cov})

        loss = {'total': total_loss, 'match': loss_match, 'pos': loss_pos, 'rot': loss_rot, 'precision': precision, 'cov': loss_cov, 'recall': recall}

        return loss
    
    def forward_scores_3dpos_cov_seq(self, out, target):
        '''
        input:
            out: dict
                scores: torch.tensor, [bs*(n+1), n+1, m+1] in log space negative
                indices: torch.tensor, [bs*(n+1), n, 1]
                cov: torch.tensor, [bs*(n+1), n, 1]
                pos: torch.tensor, [bs*(n+1), n, 3]
                scores_cam_seq: torch.tensor, [tw, bs*(n+1), m+1, m+1] in log space negative
                indices_cam_seq: torch.tensor, [tw, bs*(n+1), m, 1]
            target: dict
                pose: torch.tensor, [tw*bs*(n+1)*n, 7]
                match: torch.tensor, [tw*bs*(n+1)*n, 1] others2cam
                match_src2des: torch.tensor, [tw*bs*(n+1)*m, 1] cam last_frame to this_frame
                match_des2src: torch.tensor, [tw*bs*(n+1)*m, 1] cam this_frame to last_frame
        output:
            loss: dict
        '''

        n, m = out['scores'].size(1) - 1, out['scores'].size(2) - 1
        tw = self.args.frame_win        

        ################### match loss ##################
        recall, precision = {}, {}
        ##### others_cam #####
        label_indices_others2cam = target['match'].reshape(tw, -1, target['match'].shape[-1])[-1] # [bs*(n+1)*n, 1]
        label_indices_cam2others = target['match_cam'].reshape(tw, -1, target['match_cam'].shape[-1])[-1] # [bs*(n+1)*m, 1]
        label_indices_others_cam = {'src2des': label_indices_others2cam, 'des2src': label_indices_cam2others}
        match_others_cam = self.loss_match_func(label_indices_others_cam, out['scores'], out['indices'])
        loss_match = match_others_cam['cost_src2des'] + match_others_cam['cost_des2src']
        recall['others2cam'], precision['others2cam'] = match_others_cam['recall'], match_others_cam['precision']
        # ##### cam2cam seq #####
        label_indices_src2des = target['match_src2des'].reshape(tw, -1, target['match_src2des'].shape[-1]) # [tw, bs*(n+1)*m, 1]
        label_indices_des2src = target['match_des2src'].reshape(tw, -1, target['match_des2src'].shape[-1]) # [tw, bs*(n+1)*m, 1]
        r, p = 0, 0
        for i in range(tw-1):
            label_indices_cam2cam = {'src2des': label_indices_src2des[i], 'des2src': label_indices_des2src[i]}
            match_cam2cam = self.loss_match_func(label_indices_cam2cam, out['scores_cam_seq'][i], out['indices_cam_seq'][i])
            loss_match = loss_match + match_cam2cam['cost_src2des'] + match_cam2cam['cost_des2src']
            r, p = r + match_cam2cam['recall'], p + match_cam2cam['precision']
        recall['cam2cam'], precision['cam2cam'] = r/(tw-1), p/(tw-1)
        
        ################### pose loss ##################
        label_pose = target['pose'].reshape(tw, -1, target['pose'].shape[-1])[-1] # [bs*(n+1)*n, 7]
        out_pos = out['pos'].flatten(0,1) # [bs*(n+1)*n, 3]
        out_cov = out['cov'].flatten(0,1) # [bs*(n+1)*n, 1]
        # for all pose
        loss_pos = F.mse_loss(out_pos[:, :3], label_pose[:, :3]) * 3
        loss_rot = torch.tensor(0.0).to(loss_pos.device)
        loss_pose = loss_pos

        ################### cov loss ##################
        loss_cov = self.loss_cov_func(label_pose[:, :3], out_pos, out_cov, match_others_cam['pred_valid_mask'], dim=3)

        ################ final loss ################
        total_loss = self._combine_total({'match': loss_match, 'pos': loss_pose, 'cov': loss_cov})

        loss = {'total': total_loss, 'match': loss_match, 'pos': loss_pos, 'rot': loss_rot, 'cov': loss_cov, 
                'precision_others2cam': precision['others2cam'], 'precision_cam2cam': precision['cam2cam'], 'recall_others2cam': recall['others2cam'], 'recall_cam2cam': recall['cam2cam']}

        return loss
    
    def forward_range(self, out, target):
        '''
        input:
            out: dict
                filter_range: torch.tensor, [bs*(n+1)*n, 1]
                range_cov: torch.tensor, [bs*(n+1)*n, 1]
            target: dict
                range: torch.tensor, [bs*(n+1)*n, 1] 
        output:
            loss: dict
        '''
        loss_range = F.mse_loss(out['filter_range'], target['range'])
        loss_cov = self.loss_cov_func(target['range'], out['filter_range'], out['range_cov'], mask=None, dim=1) # [bs*(n+1)*n, 1]

        total_loss = self._combine_total({'range': loss_range, 'cov': loss_cov})

        loss = {'total': total_loss, 'range': loss_range, 'cov': loss_cov}

        return loss
    
    def forward_scores_3dpos_cov_compact(self, out, target):
        '''
        input:
            out: dict
                scores: torch.tensor, [bs*(n+1), tw*n+1, tw*m+1] in log space negative
                indices: torch.tensor, [bs*(n+1), tw*n, 1]
                pos: torch.tensor, [tw*bs*(n+1), n, 3]
                cov: torch.tensor, [tw*bs*(n+1), n, 1]
            target: dict
                pose: torch.tensor, [tw*bs*(n+1)*n, 7]
                match: torch.tensor, [tw*bs*(n+1)*n, 1] others2cam
                match_cam: torch.tensor, [tw*bs*(n+1)*m, 1] cam2others
        output:
            loss: dict
        '''
        tw, fw = self.args.frame_win, self.args.fixed_win
        n, m = int((out['scores'].size(1)-1)/tw), int((out['scores'].size(2)-1)/tw)
                
        ################### match loss ##################
        ##### others_cam #####
        label_indices_others2cam = target['match'].clone().reshape(tw, -1, n, 1)     # [tw, bs*(n+1), n, 1]
        label_indices_cam2others = target['match_cam'].clone().reshape(tw, -1, m, 1) # [tw, bs*(n+1), m, 1]
        tw_mask_dict = {} # {key: tw, value: mask[bs*(n+1)*tw*n, 1]}
        fw_src2des_mask = torch.zeros_like(label_indices_others2cam).bool() # [tw, bs*(n+1), n, 1]
        fw_des2src_mask = torch.zeros_like(label_indices_cam2others).bool() # [tw, bs*(n+1), m, 1]
        fw_src2des_mask[:fw] = True
        fw_des2src_mask[:fw] = True
        fw_src2des_mask = fw_src2des_mask.transpose(0,1).flatten(0,2) # [bs*(n+1)*tw*n, 1]
        fw_des2src_mask = fw_des2src_mask.transpose(0,1).flatten(0,2) # [bs*(n+1)*tw*m, 1]
        fw_mask_dict = {'src2des': fw_src2des_mask, 'des2src': fw_des2src_mask} # {'src2des':[bs*(n+1)*tw*n, 1], 'des2src': [bs*(n+1)*tw*m, 1]}
        for i in range(tw):
            tw_mask = torch.zeros_like(label_indices_others2cam).bool() # [tw, bs*(n+1), n, 1]
            tw_mask[i] = True
            tw_mask_dict[i] = tw_mask.transpose(0,1).flatten(0,2)

        for i in range(1, tw):
            invalid_mask_others2cam = label_indices_others2cam[i] == -1 # [bs*(n+1), n, 1]
            invalid_mask_cam2others = label_indices_cam2others[i] == -1 # [bs*(n+1), m, 1]
            plus_others2cam = torch.zeros_like(label_indices_others2cam[i]) # [bs*(n+1), n, 1]
            plus_others2cam[~invalid_mask_others2cam] = m*i
            plus_cam2others = torch.zeros_like(label_indices_cam2others[i]) # [bs*(n+1), m, 1]
            plus_cam2others[~invalid_mask_cam2others] = n*i

            label_indices_others2cam[i] = label_indices_others2cam[i] + plus_others2cam
            label_indices_cam2others[i] = label_indices_cam2others[i] + plus_cam2others

        label_indices_others2cam = label_indices_others2cam.transpose(0,1).flatten(0,2) # [bs*(n+1)*tw*n, 1]
        label_indices_cam2others = label_indices_cam2others.transpose(0,1).flatten(0,2) # [bs*(n+1)*tw*m, 1]
        label_indices_others_cam = {'src2des': label_indices_others2cam, 'des2src': label_indices_cam2others}

        match_others_cam_last = self.loss_match_func(label_indices_others_cam, out['scores'], out['indices'], mask=tw_mask_dict[tw-1], fw_mask_dict=fw_mask_dict)
        match_others_cam = self.loss_match_func(label_indices_others_cam, out['scores'], out['indices'], mask=None, fw_mask_dict=fw_mask_dict)
        loss_match = match_others_cam['cost_src2des'] + match_others_cam['cost_des2src']
        recall, precision = match_others_cam['recall'], match_others_cam['precision']
        recall_last, precision_last = match_others_cam_last['recall'], match_others_cam_last['precision']

        ################### pose loss ##################
        label_pose = target['pose'] # [tw*bs*(n+1)*n, 7]
        out_pos = out['pos'].flatten(0,1) # [tw*bs*(n+1)*n, 3]
        out_cov = out['cov'].flatten(0,1) # [tw*bs*(n+1)*n, 1]
        # for all pose
        loss_pos = F.mse_loss(out_pos[:, :3], label_pose[:, :3]) * 3
        loss_rot = torch.tensor(0.0).to(loss_pos.device)

        ################### cov loss ##################
        pred_valid_mask = match_others_cam['pred_valid_mask'].reshape(-1, tw, n).transpose(0,1).flatten() # [tw*bs*(n+1)*n]
        loss_cov = self.loss_cov_func(label_pose[:, :3], out_pos, out_cov, pred_valid_mask, dim=3)

        ################ final loss ################
        total_loss = self._combine_total({'match': loss_match, 'cov': loss_cov})

        loss = {'total': total_loss, 'match': loss_match, 'pos': loss_pos, 'rot': loss_rot, 'precision': precision, 'cov': loss_cov, 'recall': recall, 'precision_last':precision_last, 'recall_last':recall_last}

        return loss
    
    def forward_match_6dpose_compact(self, out, target):
        '''
        input:
            out: dict
                scores: torch.tensor, [bs*(n+1), tw*n+1, tw*m+1] in log space negative
                indices: torch.tensor, [bs*(n+1), tw*n, 1]
                pos: torch.tensor, [tw*bs*(n+1), n, 3] from front end
                cov: torch.tensor, [tw*bs*(n+1), n, 1]
                pose: torch.tensor, [tw*bs, n+1, n, 7] from back end
            target: dict
                pose: torch.tensor, [tw*bs*(n+1)*n, 7]
                match: torch.tensor, [tw*bs*(n+1)*n, 1] others2cam
                match_cam: torch.tensor, [tw*bs*(n+1)*m, 1] cam2others
        output:
            loss: dict
        '''
        tw, fw = self.args.frame_win, self.args.fixed_win
        n, m = int((out['scores'].size(1)-1)/tw), int((out['scores'].size(2)-1)/tw)
                
        ################### match loss ##################
        ##### others_cam #####
        label_indices_others2cam = target['match'].clone().reshape(tw, -1, n, 1)     # [tw, bs*(n+1), n, 1]
        label_indices_cam2others = target['match_cam'].clone().reshape(tw, -1, m, 1) # [tw, bs*(n+1), m, 1]
        tw_mask_dict = {} # {key: tw, value: mask[bs*(n+1)*tw*n, 1]}
        fw_src2des_mask = torch.zeros_like(label_indices_others2cam).bool() # [tw, bs*(n+1), n, 1]
        fw_des2src_mask = torch.zeros_like(label_indices_cam2others).bool() # [tw, bs*(n+1), m, 1]
        fw_src2des_mask[:fw] = True
        fw_des2src_mask[:fw] = True
        fw_src2des_mask = fw_src2des_mask.transpose(0,1).flatten(0,2) # [bs*(n+1)*tw*n, 1]
        fw_des2src_mask = fw_des2src_mask.transpose(0,1).flatten(0,2) # [bs*(n+1)*tw*m, 1]
        fw_mask_dict = {'src2des': fw_src2des_mask, 'des2src': fw_des2src_mask} # {'src2des':[bs*(n+1)*tw*n, 1], 'des2src': [bs*(n+1)*tw*m, 1]}
        for i in range(tw):
            tw_mask = torch.zeros_like(label_indices_others2cam).bool() # [tw, bs*(n+1), n, 1]
            tw_mask[i] = True
            tw_mask_dict[i] = tw_mask.transpose(0,1).flatten(0,2)

        for i in range(1, tw):
            invalid_mask_others2cam = label_indices_others2cam[i] == -1 # [bs*(n+1), n, 1]
            invalid_mask_cam2others = label_indices_cam2others[i] == -1 # [bs*(n+1), m, 1]
            plus_others2cam = torch.zeros_like(label_indices_others2cam[i]) # [bs*(n+1), n, 1]
            plus_others2cam[~invalid_mask_others2cam] = m*i
            plus_cam2others = torch.zeros_like(label_indices_cam2others[i]) # [bs*(n+1), m, 1]
            plus_cam2others[~invalid_mask_cam2others] = n*i

            label_indices_others2cam[i] = label_indices_others2cam[i] + plus_others2cam
            label_indices_cam2others[i] = label_indices_cam2others[i] + plus_cam2others

        label_indices_others2cam = label_indices_others2cam.transpose(0,1).flatten(0,2) # [bs*(n+1)*tw*n, 1]
        label_indices_cam2others = label_indices_cam2others.transpose(0,1).flatten(0,2) # [bs*(n+1)*tw*m, 1]
        label_indices_others_cam = {'src2des': label_indices_others2cam, 'des2src': label_indices_cam2others}

        # match_others_cam = self.loss_match_func(label_indices_others_cam, out['scores'], out['indices'], mask=tw_mask_dict[tw-1], fw_mask_dict=fw_mask_dict)
        match_others_cam = self.loss_match_func(label_indices_others_cam, out['scores'], out['indices'], mask=None, fw_mask_dict=fw_mask_dict)
        loss_match = match_others_cam['cost_src2des'] + match_others_cam['cost_des2src']
        recall, precision = match_others_cam['recall'], match_others_cam['precision']

        ################### pose loss ##################
        label_pose = target['pose'] # [tw*bs*(n+1)*n, 7]
        out_pos = out['pos'].flatten(0,1) # [tw*bs*(n+1)*n, 3]
        out_cov = out['cov'].flatten(0,1) # [tw*bs*(n+1)*n, 1]
        out_pose = out['pose'].reshape(-1, out['pose'].size(-1)) # [tw*bs*(n+1)*n, 7]
        # for all pose
        loss_pos = F.mse_loss(out_pose[:, :3], label_pose[:, :3]) * 3
        out_rot = utils.keep_w_positive(out_pose[:, 3:7])
        label_rot = utils.keep_w_positive(label_pose[:, 3:7])
        loss_rot = F.mse_loss(out_rot, label_rot)
        ################### cov loss ##################
        pred_valid_mask = match_others_cam['pred_valid_mask'].reshape(-1, tw, n).transpose(0,1).flatten() # [tw*bs*(n+1)*n]
        loss_cov = self.loss_cov_func(label_pose, out_pos, out_cov, pred_valid_mask)

        ################ final loss ################
        total_loss = self._combine_total({'match': loss_match, 'pos': loss_pos, 'rot': loss_rot, 'cov': loss_cov})
        
        loss = {'total': total_loss, 'match': loss_match, 'pos': loss_pos, 'rot': loss_rot, 'precision': precision, 'cov': loss_cov, 'recall': recall}

        return loss
    
    def forward_match_6dpose_compact_egat(self, out, target):
        '''
        input:
            out: dict
                scores: torch.tensor, [bs*(n+1), tw*n+1, tw*m+1] in log space negative
                indices: torch.tensor, [bs*(n+1), tw*n, 1]
                pos: torch.tensor, [tw*bs*(n+1), n, 3] from front end
                cov: torch.tensor, [tw*bs*(n+1), n, 1]
                pose: torch.tensor, [bs, n+1, n, 7] from back end. Note: only the last frame
            target: dict
                pose: torch.tensor, [tw*bs*(n+1)*n, 7]
                match: torch.tensor, [tw*bs*(n+1)*n, 1] others2cam
                match_cam: torch.tensor, [tw*bs*(n+1)*m, 1] cam2others
        output:
            loss: dict
        '''
        tw, fw = self.args.frame_win, self.args.fixed_win
        n, m = int((out['scores'].size(1)-1)/tw), int((out['scores'].size(2)-1)/tw)
                
        ################### match loss ##################
        ##### others_cam #####
        label_indices_others2cam = target['match'].clone().reshape(tw, -1, n, 1)     # [tw, bs*(n+1), n, 1]
        label_indices_cam2others = target['match_cam'].clone().reshape(tw, -1, m, 1) # [tw, bs*(n+1), m, 1]
        tw_mask_dict = {} # {key: tw, value: mask[bs*(n+1)*tw*n, 1]}
        fw_src2des_mask = torch.zeros_like(label_indices_others2cam).bool() # [tw, bs*(n+1), n, 1]
        fw_des2src_mask = torch.zeros_like(label_indices_cam2others).bool() # [tw, bs*(n+1), m, 1]
        fw_src2des_mask[:fw] = True
        fw_des2src_mask[:fw] = True
        fw_src2des_mask = fw_src2des_mask.transpose(0,1).flatten(0,2) # [bs*(n+1)*tw*n, 1]
        fw_des2src_mask = fw_des2src_mask.transpose(0,1).flatten(0,2) # [bs*(n+1)*tw*m, 1]
        fw_mask_dict = {'src2des': fw_src2des_mask, 'des2src': fw_des2src_mask} # {'src2des':[bs*(n+1)*tw*n, 1], 'des2src': [bs*(n+1)*tw*m, 1]}
        for i in range(tw):
            tw_mask = torch.zeros_like(label_indices_others2cam).bool() # [tw, bs*(n+1), n, 1]
            tw_mask[i] = True
            tw_mask_dict[i] = tw_mask.transpose(0,1).flatten(0,2)

        for i in range(1, tw):
            invalid_mask_others2cam = label_indices_others2cam[i] == -1 # [bs*(n+1), n, 1]
            invalid_mask_cam2others = label_indices_cam2others[i] == -1 # [bs*(n+1), m, 1]
            plus_others2cam = torch.zeros_like(label_indices_others2cam[i]) # [bs*(n+1), n, 1]
            plus_others2cam[~invalid_mask_others2cam] = m*i
            plus_cam2others = torch.zeros_like(label_indices_cam2others[i]) # [bs*(n+1), m, 1]
            plus_cam2others[~invalid_mask_cam2others] = n*i

            label_indices_others2cam[i] = label_indices_others2cam[i] + plus_others2cam
            label_indices_cam2others[i] = label_indices_cam2others[i] + plus_cam2others

        label_indices_others2cam = label_indices_others2cam.transpose(0,1).flatten(0,2) # [bs*(n+1)*tw*n, 1]
        label_indices_cam2others = label_indices_cam2others.transpose(0,1).flatten(0,2) # [bs*(n+1)*tw*m, 1]
        label_indices_others_cam = {'src2des': label_indices_others2cam, 'des2src': label_indices_cam2others}

        # match_others_cam = self.loss_match_func(label_indices_others_cam, out['scores'], out['indices'], mask=tw_mask_dict[tw-1], fw_mask_dict=fw_mask_dict)
        match_others_cam = self.loss_match_func(label_indices_others_cam, out['scores'], out['indices'], mask=None, fw_mask_dict=fw_mask_dict)
        loss_match = match_others_cam['cost_src2des'] + match_others_cam['cost_des2src']
        recall, precision = match_others_cam['recall'], match_others_cam['precision']

        ################### pose loss ##################
        # label_pose = target['pose'] # [tw*bs*(n+1)*n, 7] use all frame
        label_pose = target['pose'].reshape(tw, -1, 7)[-1] # [bs*(n+1)*n, 7] use the last frame
        out_pos = out['pos'].flatten(0,1) # [tw*bs*(n+1)*n, 3]
        out_cov = out['cov'].flatten(0,1) # [tw*bs*(n+1)*n, 1]
        out_pose = out['pose'].reshape(-1, out['pose'].size(-1)) # [bs*(n+1)*n, 7]
        # out_pose = out['pose'].reshape(tw, -1, 7)[-1] # use the last frame
        # for all pose
        loss_pos = F.mse_loss(out_pose[:, :3], label_pose[:, :3]) * 3
        out_rot = utils.keep_w_positive(out_pose[:, 3:7])
        label_rot = utils.keep_w_positive(label_pose[:, 3:7])
        loss_rot = F.mse_loss(out_rot, label_rot)
        # ################### cov loss ##################
        # pred_valid_mask = match_others_cam['pred_valid_mask'].reshape(-1, tw, n).transpose(0,1).flatten() # [tw*bs*(n+1)*n]
        # loss_cov = self.loss_cov_func(label_pose, out_pos, out_cov, pred_valid_mask)
        loss_cov = torch.tensor(0.0).to(loss_pos.device)

        ################ final loss ################
        # total_loss = loss_pose
        total_loss = self._combine_total({'match': loss_match, 'pos': loss_pos, 'rot': loss_rot, 'cov': loss_cov})
        
        loss = {'total': total_loss, 'match': loss_match, 'pos': loss_pos, 'rot': loss_rot, 'precision': precision, 'cov': loss_cov, 'recall': recall}

        return loss

    def forward_match_6dpose(self, out, target):
        ########### extract data from out and target ###########
        out_scores = out['scores'] # [bs*(n+1), n+1, m+1] in log space negative
        out_match = out['indices'] # [bs*(n+1), n, 1]
        out_cov = out['cov'] # [bs*(n+1), n, 1]
        out_pos = out['pos'] # [bs*(n+1), n, 3] from front end
        out_pose = out['pose'] # [bs, n+1, n, 7] from back end
        n = out_scores.size(1) - 1
        m = out_scores.size(2) - 1

        label_match = target['match'] # [bs*(n+1)*n, 1]
        label_pose = target['pose'] # [bs*(n+1)*n, 7]
        out_prob = out_scores[:, :-1, :].flatten(0, 1) # [bs*(n+1)*n, m+1]
        out_cov = out_cov.flatten(0, 1) # [bs*(n+1)*n, 1]
        out_pose = out_pose.reshape(-1, out_pose.size(-1)) # [bs*(n+1)*n, 7]
        out_pos = out_pos.reshape(-1, out_pos.size(-1)) # [bs*(n+1)*n, 3]
        
        ################### match loss ##################
        valid_mask = label_match != -1 # [bs*(n+1)*n, 1]
        valid_mask = valid_mask.flatten() # [bs*(n+1)*n]
        label_match_tmp = label_match.clone()
        label_match_tmp[~valid_mask, :] = m
        match_cost = -torch.gather(out_prob, 1, label_match_tmp) # [bs*(n+1)*n, 1]
        loss_match = match_cost.mean()

        ################# match results ################
        out_match = out_match.flatten(0,1) # [bs*(n+1)*n, 1]
        match_success = out_match == label_match # [bs*(n+1)*n, 1]
        valid_mask2 = (out_match > -1).flatten() # [bs*(n+1)*n]
        # calculate recall
        if valid_mask.any():
            valid_match_success = match_success[valid_mask, :]
            recall = valid_match_success.float().mean()
        else:
            recall = torch.tensor(1.0).to(loss_match.device)
        # calculate precision
        if valid_mask2.any():
            valid_match_success2 = match_success[valid_mask2, :] # [valid_num2, 1]
            precision = valid_match_success2.float().mean()
        else:
            precision = torch.tensor(1.0).to(loss_match.device) 

        ################### pose loss ##################
        # for all pose
        loss_pos = F.mse_loss(out_pose[:, 0:3], label_pose[:, 0:3]) * 3
        out_rot = utils.keep_w_positive(out_pose[:, 3:7])
        label_rot = utils.keep_w_positive(label_pose[:, 3:7])
        loss_rot = F.mse_loss(out_rot, label_rot)

        ################### cov loss ##################
        pos_diff = (out_pos[valid_mask2, :3] - label_pose[valid_mask2, :3]).unsqueeze(-1) # [valid_num2, 3, 1]
        # calculate cov mat according to cov
        cov = out_cov[valid_mask2, :].repeat(1, 3) # [valid_num2, 3]
        cov_mat = torch.diag_embed(cov) # [valid_mask2, 3, 3]
        info_mat = torch.inverse(cov_mat) # [valid_mask2, 3, 3]
        det_cov = torch.det(cov_mat).unsqueeze(-1) # [valid_mask2, 1]
        loss_out_cov = torch.log(det_cov)
        loss_weighted_pos = torch.matmul(torch.matmul(pos_diff.transpose(1, 2), info_mat), pos_diff).squeeze(-1) # [valid_mask2, 1]
        loss_cov = (loss_weighted_pos + loss_out_cov * self.loss_weights['cov_det']).mean()

        ################ final loss ################
        # total_loss = loss_pose
        total_loss = self._combine_total({'match': loss_match, 'pos': loss_pos, 'rot': loss_rot, 'cov': loss_cov})
        
        loss = {'total': total_loss, 'match': loss_match, 'pos': loss_pos, 'rot': loss_rot, 'precision': precision, 'cov': loss_cov, 'recall': recall}

        return loss
    
    def forward_handcrafted(self, out, target):
        label_match = target['match']
        label_pos = target['pose']
        out_pos = out['pos'].flatten(0, 1) # [bs*(n+1)*n, 3]
        loss_pos = F.mse_loss(out_pos[:, :3], label_pos[:, :3]) * 3
        valid_mask = label_match != -1 # [batch_size*match_num, 1]
        valid_mask = valid_mask.flatten() # [batch_size*match_num]
        out_match = out['indices'].flatten(0,1) # [bs*(n+1)*n, 1]
        match_success = out_match == label_match # [bs*(n+1)*n, 1]
        valid_mask2 = (out_match > -1).flatten() # [bs*(n+1)*n]
        # calculate recall
        if valid_mask.any():
            valid_match_success = match_success[valid_mask, :]
            recall = valid_match_success.float().mean()
        else:
            recall = torch.tensor(1.0).to(loss_pos.device)

        # calculate precision
        if valid_mask2.any():
            valid_match_success2 = match_success[valid_mask2, :] # [valid_num2, 1]
            precision = valid_match_success2.float().mean()
        else:
            precision = torch.tensor(1.0).to(loss_pos.device) 

        loss_match = loss_rot = loss_cov = torch.tensor(0.0).to(loss_pos.device)
        loss = self._combine_total({'pos': loss_pos})
        loss = {'total': loss, 'match': loss_match, 'pos': loss_pos, 'precision': precision, 'recall': recall, 'rot': loss_rot, 'cov': loss_cov}
        return loss

    def forward_ref(self, out, target):
        '''
        input:
            out: dict
                moving_pos: list[ torch.tensor, [num_m, 3] ]
                moving_cov: torch.tensor, [num_m, 3]
                base_pose: list[ torch.tensor, [num_b, 7] ]
                base_cov: torch.tensor, [num_b, 6]
                dis_loss: Optional, torch.tensor
                dis_cov: Optional, torch.tensor, [num_d, 1]
                dis_diff: Optional, torch.tensor, [num_d, 1]
            target: dict
                moving_pos: torch.tensor, [num_m, 3]
                base_pose: torch.tensor, [num_b, 7]
                multi_tag: torch.tensor, [num_b]
        output:
            loss: dict
        '''
        total_loss = torch.tensor(0.0).to(out['moving_pos'][0].device)
        for moving_pos, base_pose in zip(out['moving_pos'], out['base_pose']):
            loss_tagpos = F.mse_loss(moving_pos[:,:3], target['moving_pos'][:,:3]) * 3
            loss_pos = F.mse_loss(base_pose[:,:3], target['base_pose'][:,:3]) * 3
            total_loss += self._combine_total({'pos': loss_pos, 'tagpos': loss_tagpos})
            if target['multi_tag'].any():
                out_rot = utils.keep_w_positive(base_pose[target['multi_tag'], 3:7])
                label_rot = utils.keep_w_positive(target['base_pose'][target['multi_tag'], 3:7])
                loss_rot = F.mse_loss(out_rot, label_rot)
                total_loss += self._combine_total({'rot': loss_rot})
            else:
                loss_rot = torch.tensor(5.0).to(loss_tagpos.device)

        loss_cov = self.loss_cov_func(target['moving_pos'], out['moving_pos'][-1], out['moving_cov'], dim=3) + \
                    self.loss_cov_func(target['base_pose'][:, :3], out['base_pose'][-1][:, :3], out['base_cov'][:, :3], dim=3) + \
                    self.loss_cov_func(target['base_pose'][:, 3:], out['base_pose'][-1][:, 3:], out['base_cov'][:, 3:], dim=3, mask=target['multi_tag'])

        total_loss += self._combine_total({'cov': loss_cov})

        loss = {'total': total_loss, 'tagpos': loss_tagpos, 'pos': loss_pos, 'rot': loss_rot, 'cov': loss_cov}

        return loss
    
    def forward_ref_pgo(self, out, target):
        '''
        input:
            out: dict
                moving_pos: list[ torch.tensor, [num_m, 3] ]
                moving_cov: torch.tensor, [num_m, 3]
                base_pose: list[ torch.tensor, [num_b, 7] ]
                base_cov: torch.tensor, [num_b, 6]
                dis_loss: Optional, torch.tensor
                dis_cov: Optional, torch.tensor, [num_d, 1]
                dis_diff: Optional, torch.tensor, [num_d, 1]
            target: dict
                moving_pos: torch.tensor, [num_m, 3]
                base_pose: torch.tensor, [num_b, 7]
                multi_tag: torch.tensor, [num_b]
        output:
            loss: dict
        '''
        total_loss = torch.tensor(0.0).to(out['moving_pos'][0].device)
        for moving_pos, base_pose in zip(out['moving_pos'], out['base_pose']):
            loss_tagpos = F.mse_loss(moving_pos[:,:3], target['moving_pos'][:,:3]) * 3
            loss_pos = F.mse_loss(base_pose[:,:3], target['base_pose'][:,:3]) * 3
            total_loss += self._combine_total({'pos': loss_pos, 'tagpos': loss_tagpos})
            # total_loss += loss_tagpos # Use this when single-robot-single-tag!
            if target['multi_tag'].any():
                out_rot = utils.keep_w_positive(base_pose[target['multi_tag'], 3:7])
                label_rot = utils.keep_w_positive(target['base_pose'][target['multi_tag'], 3:7])
                loss_rot = F.mse_loss(out_rot, label_rot)
                total_loss += self._combine_total({'rot': loss_rot})
            else:
                loss_rot = torch.tensor(5.0).to(loss_tagpos.device)

        if not target['multi_tag'].any(): # Supervise the last layer(PGO rot), but if single-robot-single-tag, don't use this!
            out_rot = utils.keep_w_positive(out['base_pose'][-1][:, 3:7])
            label_rot = utils.keep_w_positive(target['base_pose'][:, 3:7])
            loss_rot = F.mse_loss(out_rot, label_rot)
            total_loss += self._combine_total({'rot': loss_rot})
        
        # loss_cov = self.loss_cov_func(target['moving_pos'], out['moving_pos'][-1], out['moving_cov'], dim=3) + \
        #             self.loss_cov_func(target['base_pose'][:, :3], out['base_pose'][-1][:, :3], out['base_cov'][:, :3], dim=3) + \
        #             self.loss_cov_func(target['base_pose'][:, 3:], out['base_pose'][-1][:, 3:], out['base_cov'][:, 3:], dim=3, mask=target['multi_tag'])
        loss_cov = torch.tensor(0.0, device=total_loss.device)
        total_loss += self._combine_total({'cov': loss_cov})

        loss = {'total': total_loss, 'tagpos': loss_tagpos, 'pos': loss_pos, 'rot': loss_rot, 'cov': loss_cov}

        return loss
    
    ###############################################################################################

    def loss_match_func(self, label_indices, out_scores, out_indices, mask=None, fw_mask_dict=None):
        '''
        input:
            label_indices: dict
                src2des: torch.tensor, [bs*n, 1]
                des2src: torch.tensor, [bs*m, 1]
            out_scores: torch.tensor, [bs, n+1, m+1]
            out_indices: torch.tensor, [bs, n, 1]
            mask: Optional, torch.tensor, [bs*n, 1]
            fw_mask_dict: Optional, dict
                src2des: torch.tensor, [bs*n, 1]
                des2src: torch.tensor, [bs*m, 1]
        output:
            match_res: dict
        '''
        n, m = out_scores.size(1) - 1, out_scores.size(2) - 1
        ##### src2des #####
        label_indices_s2d = label_indices['src2des'].clone()
        label_valid_mask_s2d = (label_indices_s2d != -1).flatten() # [bs*n]
        label_indices_s2d[~label_valid_mask_s2d, :] = m
        scores_s2d = out_scores[:, :-1, :].flatten(0, 1) # [bs*n, m+1]
        cost_s2d = -torch.gather(scores_s2d, 1, label_indices_s2d) # [bs*n, 1]
        ##### des2src #####
        label_indices_d2s = label_indices['des2src'].clone().reshape(-1, m).unsqueeze(1) # [bs, 1, m]
        label_valid_mask_d2s = label_indices_d2s != -1
        label_indices_d2s[~label_valid_mask_d2s] = n
        scores_d2s = out_scores[:, :, :-1] # [bs, n+1, m]
        cost_d2s = -torch.gather(scores_d2s, 1, label_indices_d2s) # [bs, 1, m]

        ################# match recall & precision ################
        out_indices = out_indices.flatten(0,1) # [bs*n, 1]
        indices_success = out_indices == label_indices['src2des'] # [bs*n, 1]
        pred_valid_mask_s2d = (out_indices != -1).flatten() # [bs*n]

        if isinstance(fw_mask_dict, dict):
            cost_s2d = cost_s2d[~fw_mask_dict['src2des']]
            cost_d2s = cost_d2s.reshape(fw_mask_dict['des2src'].shape)[~fw_mask_dict['des2src']]
            label_valid_mask_s2d = label_valid_mask_s2d & (~fw_mask_dict['src2des']).flatten()
            pred_valid_mask_s2d = pred_valid_mask_s2d & (~fw_mask_dict['src2des']).flatten()

        if mask is not None:
            label_valid_mask_s2d = label_valid_mask_s2d & mask.flatten()
            pred_valid_mask_s2d = pred_valid_mask_s2d & mask.flatten()

        ### recall ###
        if label_valid_mask_s2d.any():
            valid_match_success = indices_success[label_valid_mask_s2d, :]
            recall = valid_match_success.float().mean()
        else:
            recall = torch.tensor(1.0).to(out_scores.device)
        ### precision ###
        if pred_valid_mask_s2d.any():
            valid_match_success = indices_success[pred_valid_mask_s2d, :]
            precision = valid_match_success.float().mean()
        else:
            precision = torch.tensor(1.0).to(out_scores.device)

        match_res = {'cost_src2des': cost_s2d.mean(), 'cost_des2src': cost_d2s.mean(), 'precision': precision, 'recall': recall,
                     'label_valid_mask': label_valid_mask_s2d, 'pred_valid_mask': pred_valid_mask_s2d}
        return match_res
    
    def loss_cov_func(self, label, pred, cov, mask=None, dim=3, det_weight=None):
        '''
        calculate the covariance loss of predicted [ dis(dim=1) or pos(dim=3) or quat(dim=3) or pose(dim=6) ]
        input:
            label: torch.tensor, [bs, 1 or 3 or 4 or 7]
            pred: torch.tensor, [bs, 1 or 3 or 4 or 7]
            cov: torch.tensor, [bs, 1 or 3 or 6]
            mask: torch.tensor, [bs]
            dim: int, 1 or 3 or 4 or 6
        output:
            loss_cov: torch.tensor
        '''
        if det_weight is None:
            det_weight = self.loss_weights['cov_det']
        if mask is None:
            mask = torch.ones_like(label[:, 0]).bool() # [bs]
        if (~mask).all():
            return torch.tensor(5.0).to(label.device)
        assert min(label.shape[-1], pred.shape[-1]) >= dim

        if dim == 1:
            diff = (pred[mask, :] - label[mask, :]).unsqueeze(-1) # [valid_num, 1, 1]
        elif dim == 3:
            if label.shape[-1] == 3: # pos
                diff = (pred[mask, :3] - label[mask, :3]).unsqueeze(-1) # [valid_num, 3, 1]
            elif label.shape[-1] == 4: # quat
                quat_diff = utils.batched_quat_diff_torch(pred[mask, :4], label[mask, :4], w_first=False) # [valid_num, 4]
                euler_diff = utils.batched_quat_to_euler_torch(quat_diff, w_first=True) # [valid_num, 3]
                diff = euler_diff.unsqueeze(-1) # [valid_num, 3, 1]
            else:
                raise ValueError("label shape is not valid in dim=3")
        elif dim == 6:
            diff = (pred[mask, :3] - label[mask, :3]).unsqueeze(-1) # [valid_num, 3, 1]
            quat_diff = utils.batched_quat_diff_torch(pred[mask, 3:7], label[mask, 3:7], w_first=False) # [valid_num, 4]
            euler_diff = utils.batched_quat_to_euler_torch(quat_diff, w_first=True) # [valid_num, 3]
            diff = torch.cat((diff, euler_diff.unsqueeze(-1)), dim=1) # [valid_num, 6, 1]
        else:
            raise ValueError("dim is not valid")

        if cov.shape[1] == dim:
            cov = cov[mask, :] # [valid_num, dim]
        elif cov.shape[1] == 1:
            cov = cov[mask, :].repeat(1, dim) # [valid_num, dim]
        else:
            raise ValueError("out_cov shape is not valid")
        cov_mat = torch.diag_embed(cov) # [valid_num, dim, dim]
        info_mat = torch.inverse(cov_mat) # [valid_num, dim, dim]
        det_cov = torch.det(cov_mat).unsqueeze(-1) # [valid_num, 1]
        loss_out_cov = torch.log(det_cov)
        loss_weighted_diff = torch.matmul(torch.matmul(diff.transpose(1, 2), info_mat), diff).squeeze(-1) # [valid_num, 1]
        loss_cov = (loss_weighted_diff + loss_out_cov*det_weight).mean()
        return loss_cov
