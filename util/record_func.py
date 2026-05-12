import os
import torch
from util.utils import set_pose_accuracy, ind2id, id2ind
import pandas as pd
import random
import transforms3d as tfs
import numpy as np
import math

class RecordGraph:
    def __init__(self, wrt_folder, graph_ind=-1, allow_moving_self_edge=True, use_uwb_seq=False, max_tag_num=4, add_noise=True):

        self.allow_moving_self_edge = allow_moving_self_edge
        self.use_uwb_seq = use_uwb_seq
        self.max_tag_num = max_tag_num
        self.add_noise = add_noise
        # self.noise_list = [0.2, 0.2, 0.2, 0.5/180*math.pi, 0.5/180*math.pi, 0.5/180*math.pi] # [pos_noise, pos_noise, pos_noise, rot_noise, rot_noise, rot_noise]
        self.noise_list = [0.1, 0.1, 0.1, 0.2/180*math.pi, 0.2/180*math.pi, 0.2/180*math.pi] # [pos_noise, pos_noise, pos_noise, rot_noise, rot_noise, rot_noise]

        if not os.path.exists(wrt_folder):
            os.makedirs(wrt_folder)

        self.graphs_file = os.path.join(wrt_folder, "graphs.csv")
        self.nodes_fixed_file = os.path.join(wrt_folder, "nodes_fixed.csv")
        self.nodes_moving_file = os.path.join(wrt_folder, "nodes_moving.csv")
        self.nodes_base_file = os.path.join(wrt_folder, 'nodes_base.csv')
        self.nodes_ref_file = os.path.join(wrt_folder, 'nodes_ref.csv')
        self.edges_fixed2moving_file = os.path.join(wrt_folder, "edges_fixed2moving.csv")
        self.edges_moving2fixed_file = os.path.join(wrt_folder, "edges_moving2fixed.csv")
        self.edges_moving2moving_file = os.path.join(wrt_folder, "edges_moving2moving.csv")
        self.edges_moving2base_file = os.path.join(wrt_folder, "edges_moving2base.csv")
        self.edges_ref2base_file = os.path.join(wrt_folder, "edges_ref2base.csv")
        self.graph_ind = graph_ind
        ##### graph level #####
        self.g_graphid_list = []
        self.g_timestamp_list = []
        self.g_refid_list = []
        self.g_world_pose_delta_list = []
        self.g_local2map_list = []
        ##### node level #####
        # fixed node
        self.nf_graphid_list = []
        self.nf_nodeid_list = []
        self.nf_feat = []
        # moving node
        self.nm_graphid_list = []
        self.nm_nodeid_list = []
        self.nm_feat = [] # [x, y, z]
        self.nm_label = [] # [x, y, z]
        self.nm_exparam = [] # [tx, ty, tz]
        self.nm_valid = [] # 0 for padding, 1 for moving tags
        # base node
        self.nb_graphid_list = []
        self.nb_nodeid_list = []
        self.nb_feat = [] # [x, y, z, qx, qy, qz, qw]
        self.nb_label = [] # [x, y, z, qx, qy, qz, qw]
        self.nb_valid = [] # 0 for padding / ref frame, 1 for moving base
        self.nb_multi_tag = [] # 0 for 1-tag, 1 for multi-tag
        # ref node
        self.nr_graphid_list = []
        self.nr_nodeid_list = []
        self.nr_feat = [] # [x, y, z, qx, qy, qz, qw]
        ##### edge level #####
        # fixed2moving edge
        self.ef2m_graphid_list = []
        self.ef2m_srcid_list = []
        self.ef2m_dstid_list = []
        self.ef2m_eid_list = [] # [src_id, dst_id] for each edge
        self.ef2m_feat = [] # noisy_dis_seq[d1, d2, ..., dN]
        self.ef2m_label = [] # [gt_dis]
        # moving2fixed edge
        self.em2f_graphid_list = []
        self.em2f_srcid_list = []
        self.em2f_dstid_list = []
        self.em2f_eid_list = [] # [src_id, dst_id] for each edge
        self.em2f_feat = [] # noisy_dis_seq[d1, d2, ..., dN]
        self.em2f_label = [] # [gt_dis]
        # moving2moving edge
        self.em2m_graphid_list = []
        self.em2m_srcid_list = []
        self.em2m_dstid_list = []
        self.em2m_eid_list = [] # [src_id, dst_id] for each edge
        self.em2m_feat = [] # noisy_dis_seq[d1, d2, ..., dN]
        self.em2m_label = [] # [gt_dis]
        # moving2base edge
        self.em2b_graphid_list = []
        self.em2b_srcid_list = []
        self.em2b_dstid_list = []
        self.em2b_feat = [] # [tx, ty, tz] exparam
        # ref2base edge
        self.er2b_graphid_list = []
        self.er2b_srcid_list = []
        self.er2b_dstid_list = []
        self.er2b_feat = [] # [bx, by, bz] bearing

    def add_pose_noise(self, ori_data, noise):
        """
        input: ori_data [x, y, z, qx, qy, qz, qw]
                noise [pos_noise, pos_noise, pos_noise, rot_noise, rot_noise, rot_noise]
        output: noisy_data [x, y, z, qx, qy, qz, qw]
        """
        data = ori_data.copy()
        for i in range(3):
            data[i] += random.gauss(0, noise[i])
        ori_quat = np.array([data[6], data[3], data[4], data[5]]) # w x y z
        roll, pitch, yaw = tfs.euler.quat2euler(ori_quat, axes='sxyz')
        roll += random.gauss(0, noise[3])
        pitch += random.gauss(0, noise[4])
        yaw += random.gauss(0, noise[5])
        quat = tfs.euler.euler2quat(roll, pitch, yaw, axes='sxyz')
        data[3] = quat[1]
        data[4] = quat[2]
        data[5] = quat[3]
        data[6] = quat[0]
        return data
    
    ########################################################################################
    
    def record_graph_tolist_ref(self, pose, label_pose, uwb_range_seq, uwb_range_gt, bearing, bearing_mask, local2map, world_pose_delta, ts):
        '''
        input:
            pose: [bs, n+1, n, 7]
            label_pose: [bs, n+1, n, 7]
            uwb_range_seq: [bs, n+1, n, dis_len]
            uwb_range_gt: [bs, n+1, n, 1]
            bearing: [bs, n+1, n, 3]
            bearing_mask: [bs, n+1, n]
            local2map: [bs, n+1, 7]
            world_pose_delta: [bs, n+1, 7]
            ts: float
        '''
        bs, n = pose.shape[0], pose.shape[2]
        integer_t, decimal_t = str(ts).split('.')
        decimal_t = '1' + decimal_t # int can't start with 0
        timestamp = integer_t + ',' + decimal_t
        assert bs == 1, 'only support batch size = 1'
        zero_pose = ','.join(str(p) for p in [0.0]*6+[1.0])  # [x, y, z, qx, qy, qz, qw]
        zero_3d = ','.join(str(p) for p in [0.0]*3)
        for bi in range(bs):
            for k in range(n+1):
                self.graph_ind += 1
                self.g_graphid_list.append(self.graph_ind)
                self.g_timestamp_list.append(timestamp)
                self.g_refid_list.append(f'{k},')
                world_pose_delta_feat = ','.join(str(p) for p in set_pose_accuracy(world_pose_delta[bi, k].cpu().tolist()))
                local2map_feat = ','.join(str(p) for p in set_pose_accuracy(local2map[bi, k].cpu().tolist()))
                self.g_world_pose_delta_list.append(world_pose_delta_feat)
                self.g_local2map_list.append(local2map_feat)
                ### fix node ###
                self.nf_graphid_list.append(self.graph_ind)
                self.nf_nodeid_list.append(0)
                self.nf_feat.append(zero_3d)
                ### ref node ###
                self.nr_graphid_list.append(self.graph_ind)
                self.nr_nodeid_list.append(0)
                self.nr_feat.append(zero_pose)
                ### base node -- padding ###
                self.nb_graphid_list.append(self.graph_ind)
                self.nb_nodeid_list.append(0)
                self.nb_feat.append(zero_pose) 
                self.nb_label.append(zero_pose)
                self.nb_valid.append(0) # padding base
                self.nb_multi_tag.append(0) # padding base
                ### ref2base edge -- padding ###
                self.er2b_graphid_list.append(self.graph_ind)
                self.er2b_srcid_list.append(0)
                self.er2b_dstid_list.append(0)
                self.er2b_feat.append(zero_3d)
                ### moving node -- padding ###
                self.nm_graphid_list.append(self.graph_ind)
                self.nm_nodeid_list.append(0) # padding moving
                self.nm_feat.append(zero_3d)
                self.nm_label.append(zero_3d)
                self.nm_exparam.append(zero_3d)
                self.nm_valid.append(0) # padding moving
                for i in range(n):
                    ### moving node -- others ###
                    self.nm_graphid_list.append(self.graph_ind)
                    self.nm_nodeid_list.append(i+1)
                    pose_pred = pose[bi, k, i].cpu().tolist()
                    if self.add_noise:
                        pose_pred = self.add_pose_noise(pose_pred, self.noise_list)
                    pose_gt = label_pose[bi, k, i].cpu().tolist()
                    self.nm_feat.append(','.join(str(round(p,3)) for p in pose_pred[:3]))
                    self.nm_label.append(','.join(str(round(p,3)) for p in pose_gt[:3]))
                    self.nm_exparam.append(','.join(str(p) for p in [0.0, 0.0, 0.0]))
                    self.nm_valid.append(1) # moving tag
                    ### base node -- others the same as moving node ###
                    self.nb_graphid_list.append(self.graph_ind)
                    self.nb_nodeid_list.append(i+1)
                    base_feat = ','.join(str(p) for p in set_pose_accuracy(pose_pred))
                    self.nb_feat.append(base_feat)
                    gt_feat = ','.join(str(p) for p in set_pose_accuracy(pose_gt))
                    self.nb_label.append(gt_feat) # [x, y, z, qx, qy, qz, qw]
                    self.nb_valid.append(1) # moving base
                    self.nb_multi_tag.append(0) # 1-tag
                    ### moving2base edge -- others tag to others base ###
                    self.em2b_graphid_list.append(self.graph_ind)
                    self.em2b_srcid_list.append(i+1)
                    self.em2b_dstid_list.append(i+1)
                    self.em2b_feat.append(zero_3d)
                    ### ref2base edge -- ref to base ###
                    if bearing_mask[bi, k, i]:
                        self.er2b_graphid_list.append(self.graph_ind)
                        self.er2b_srcid_list.append(0)
                        self.er2b_dstid_list.append(i+1)
                        bearing_feat = ','.join(str(round(b,6)) for b in bearing[bi, k, i].cpu().tolist())
                        self.er2b_feat.append(bearing_feat)
                    ### moving2fixed edge -- others to ref ###
                    src_embed_id = ind2id(k, i)*self.max_tag_num + 0 # tag_num is 1
                    dst_embed_id = k*self.max_tag_num + 0
                    eid_feat = str(src_embed_id) + ',' + str(dst_embed_id)
                    if self.use_uwb_seq:
                        dis = uwb_range_seq[bi, k, i].cpu().tolist()
                        dis_feat = ','.join(str(round(d,3)) for d in dis)
                    else:
                        dis = uwb_range_seq[bi, k, i, -1].item()
                        dis_feat = str(round(dis,3))+','
                    gt_dis = uwb_range_gt[bi, k, i, 0].item()
                    self.em2f_graphid_list.append(self.graph_ind)
                    self.em2f_srcid_list.append(i+1)
                    self.em2f_dstid_list.append(0)
                    self.em2f_eid_list.append(eid_feat)
                    self.em2f_feat.append(dis_feat)
                    self.em2f_label.append(str(round(gt_dis,3))+',')
                    ### fixed2moving edge -- ref to others ###
                    id_fixed = k
                    id_moving = ind2id(k, i)
                    src_embed_id = id_fixed*self.max_tag_num + 0 # tag_num is 1
                    dst_embed_id = id_moving*self.max_tag_num + 0
                    eid_feat = str(src_embed_id) + ',' + str(dst_embed_id)
                    if self.use_uwb_seq:
                        dis = uwb_range_seq[bi, id_moving, id2ind(id_moving, id_fixed)].cpu().tolist()
                        dis_feat = ','.join(str(round(d,3)) for d in dis)
                    else:
                        dis = uwb_range_seq[bi, id_moving, id2ind(id_moving, id_fixed), -1].item()
                        dis_feat = str(round(dis,3))+','
                    gt_dis = uwb_range_gt[bi, id_moving, id2ind(id_moving, id_fixed), 0].item()
                    self.ef2m_graphid_list.append(self.graph_ind)
                    self.ef2m_srcid_list.append(0)
                    self.ef2m_dstid_list.append(i+1)
                    self.ef2m_eid_list.append(eid_feat)
                    self.ef2m_feat.append(dis_feat)
                    self.ef2m_label.append(str(round(gt_dis,3))+',')

                ### moving2moving edge -- others to others ###
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        id_moving1 = ind2id(k, i)
                        id_moving2 = ind2id(k, j)
                        # i to j
                        if self.use_uwb_seq:
                            dis = uwb_range_seq[bi, id_moving2, id2ind(id_moving2, id_moving1)].cpu().tolist()
                            dis_feat = ','.join(str(round(d,3)) for d in dis)
                        else:
                            dis = uwb_range_seq[bi, id_moving2, id2ind(id_moving2, id_moving1), -1].item()
                            dis_feat = str(round(dis,3))+','                        
                        gt_dis = uwb_range_gt[bi, id_moving2, id2ind(id_moving2, id_moving1), 0].item()
                        self.em2m_graphid_list.append(self.graph_ind)
                        self.em2m_srcid_list.append(i+1)
                        self.em2m_dstid_list.append(j+1)
                        src_embed_id = id_moving1*self.max_tag_num + 0 # tag_num is 1
                        dst_embed_id = id_moving2*self.max_tag_num + 0
                        eid_feat = str(src_embed_id) + ',' + str(dst_embed_id)
                        self.em2m_eid_list.append(eid_feat)
                        self.em2m_feat.append(dis_feat)
                        self.em2m_label.append(str(round(gt_dis,3))+',')
    
    def save_to_csv_ref(self):
        dataframe_graphs = pd.DataFrame({'graph_id':self.g_graphid_list, 'timestamp':self.g_timestamp_list, 'ref_id': self.g_refid_list, 'world_pose_delta': self.g_world_pose_delta_list, 'local2map': self.g_local2map_list})
        dataframe_graphs.to_csv(self.graphs_file, index=False, sep=',')
        dataframe_node_fixed = pd.DataFrame({'graph_id':self.nf_graphid_list, 'node_id':self.nf_nodeid_list, 'feat':self.nf_feat})
        dataframe_node_fixed.to_csv(self.nodes_fixed_file, index=False, sep=',')
        dataframe_node_moving = pd.DataFrame({'graph_id':self.nm_graphid_list, 'node_id':self.nm_nodeid_list, 'feat':self.nm_feat, 'label':self.nm_label, 'exparam':self.nm_exparam, 'valid':self.nm_valid})
        dataframe_node_moving.to_csv(self.nodes_moving_file, index=False, sep=',')
        dataframe_node_base = pd.DataFrame({'graph_id':self.nb_graphid_list, 'node_id':self.nb_nodeid_list, 'feat':self.nb_feat, 'label':self.nb_label, 'valid':self.nb_valid, 'multi_tag':self.nb_multi_tag})
        dataframe_node_base.to_csv(self.nodes_base_file, index=False, sep=',')
        dataframe_node_ref = pd.DataFrame({'graph_id':self.nr_graphid_list, 'node_id':self.nr_nodeid_list, 'feat':self.nr_feat})
        dataframe_node_ref.to_csv(self.nodes_ref_file, index=False, sep=',')
        dataframe_edges_fixed2moving = pd.DataFrame({'graph_id':self.ef2m_graphid_list, 'src_id':self.ef2m_srcid_list, 'dst_id':self.ef2m_dstid_list, 'eid':self.ef2m_eid_list, 'feat':self.ef2m_feat, 'label':self.ef2m_label})
        dataframe_edges_fixed2moving.to_csv(self.edges_fixed2moving_file, index=False, sep=',')
        dataframe_edges_moving2fixed = pd.DataFrame({'graph_id':self.em2f_graphid_list, 'src_id':self.em2f_srcid_list, 'dst_id':self.em2f_dstid_list, 'eid':self.em2f_eid_list, 'feat':self.em2f_feat, 'label':self.em2f_label})
        dataframe_edges_moving2fixed.to_csv(self.edges_moving2fixed_file, index=False, sep=',')
        dataframe_edges_moving2moving = pd.DataFrame({'graph_id':self.em2m_graphid_list, 'src_id':self.em2m_srcid_list, 'dst_id':self.em2m_dstid_list, 'eid':self.em2m_eid_list, 'feat':self.em2m_feat, 'label':self.em2m_label})
        dataframe_edges_moving2moving.to_csv(self.edges_moving2moving_file, index=False, sep=',')
        dataframe_edges_moving2base = pd.DataFrame({'graph_id':self.em2b_graphid_list, 'src_id':self.em2b_srcid_list, 'dst_id':self.em2b_dstid_list, 'feat':self.em2b_feat})
        dataframe_edges_moving2base.to_csv(self.edges_moving2base_file, index=False, sep=',')
        dataframe_edges_ref2base = pd.DataFrame({'graph_id':self.er2b_graphid_list, 'src_id':self.er2b_srcid_list, 'dst_id':self.er2b_dstid_list, 'feat':self.er2b_feat})
        dataframe_edges_ref2base.to_csv(self.edges_ref2base_file, index=False, sep=',')
        print("Write CSV File Done!")