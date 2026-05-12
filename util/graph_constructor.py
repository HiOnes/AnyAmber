import os
import torch
from util.utils import set_pose_accuracy, ind2id, id2ind
import pandas as pd
import random
import transforms3d as tfs
import numpy as np
import math
import dgl

def rm_edge(g, rm_etype):
    keep_etypes = [etype for etype in g.canonical_etypes
               if etype != rm_etype]
    sub_g = dgl.edge_type_subgraph(g, keep_etypes)
    return sub_g

def replace_edges_batch(g1, g2, edge_type=('ref', 'ref2base', 'base')):
    """
    replace the edge of g1 with the edge in g2
    """
    g_update = g1.clone()
    src1, dst1 = g1.edges(etype=edge_type)
    edge_ids1 = g1.edge_ids(src1, dst1, etype=edge_type)
    g_update.remove_edges(edge_ids1, etype=edge_type)

    src2, dst2 = g2.edges(etype=edge_type)
    edge_features2 = g2.edges[edge_type].data
    g_update.add_edges(src2, dst2, etype=edge_type)
    
    for feat_name, feat_value in edge_features2.items():
        g_update.edges[edge_type].data[feat_name] = feat_value.clone()
    
    return g_update

class GraphConstructor:
    def __init__(self, use_uwb_seq=False, max_tag_num=4):
        self.use_uwb_seq = use_uwb_seq
        self.max_tag_num = max_tag_num
    
    def replace_graph_bearing(self, pose, bearing, bearing_mask, bg):
        '''
        input:
            pose: [bs, n+1, n, 7]
            bearing: [bs, n+1, n, 3]
            bearing_mask: [bs, n+1, n]
            bg: dgl.batch graph
        output:
            batch_graph: dgl.batch graph
        '''
        bs, n, device = pose.shape[0], pose.shape[2], pose.device
        assert bs == 1, 'Only support batch size = 1'
        # bg = rm_edge(bg, ('ref', 'ref2base', 'base'))
        graph_list = []
        for bi in range(bs):
            for k in range(n+1):
                ### construct graph ###
                graph = dgl.heterograph({
                    ('moving', 'moving2fixed', 'fixed'): ([], []),
                    ('fixed', 'fixed2moving', 'moving'): ([], []),
                    ('moving', 'moving2moving', 'moving'): ([], []),
                    ('moving', 'moving2base', 'base'): ([], []),
                    ('ref', 'ref2base', 'base'): ([], [])
                }).to(device)
                ### ref & base node ###
                graph.add_nodes(1, ntype='ref')
                graph.add_nodes(n+1, ntype='base')
                ### ref2base edge -- ref to others base ###
                src_ids = [0]
                dst_ids = [0]
                feats = [torch.zeros(3, device=device)]
                for i in range(n):
                    if bearing_mask[bi, k, i]:
                        src_ids.append(0)
                        dst_ids.append(i+1)
                        feats.append(bearing[bi, k, i])
                
                graph.add_edges(
                    src_ids, dst_ids, 
                    etype=('ref', 'ref2base', 'base'),
                    data={'feat': torch.stack(feats)}
                )
                graph_list.append(graph)

        batch_graph = dgl.batch(graph_list)
        bg_update = replace_edges_batch(bg, batch_graph, ('ref', 'ref2base', 'base'))

        return bg_update
    
    def construct_graph_ref(self, pose, label_pose, uwb_range_seq, uwb_range_gt, bearing, bearing_mask, 
                           local2map, world_pose_delta, ts):
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
        output:
            batch_graph: dgl.batch graph
        '''
        bs, n, device = pose.shape[0], pose.shape[2], pose.device
        assert bs == 1, 'Only support batch size = 1'
        graph_list = []
        for bi in range(bs):
            for k in range(n+1):
                ### construct graph ###
                graph = dgl.heterograph({
                    ('moving', 'moving2fixed', 'fixed'): ([], []),
                    ('fixed', 'fixed2moving', 'moving'): ([], []),
                    ('moving', 'moving2moving', 'moving'): ([], []),
                    ('moving', 'moving2base', 'base'): ([], []),
                    ('ref', 'ref2base', 'base'): ([], [])
                }).to(device)

                # # 添加图级别属性
                # graph.graph_id = torch.tensor([k])
                # graph.timestamp = torch.tensor([ts])
                # graph.ref_id = torch.tensor([k])
                # graph.world_pose_delta = world_pose_delta[bi, k].unsqueeze(0)
                # graph.local2map = local2map[bi, k].unsqueeze(0)
                
                ### fixed node ###
                graph.add_nodes(1, ntype='fixed')
                graph.nodes['fixed'].data['feat'] = torch.zeros(1, 3, device=device)  # [x, y, z]
                
                ### ref node ###
                graph.add_nodes(1, ntype='ref')
                graph.nodes['ref'].data['feat'] = torch.tensor([0.0]*6+[1.0], device=device).reshape(1,7)  # [x, y, z, qx, qy, qz, qw]
                
                ### moving nodes & base nodes ###
                # 每个图有n+1个节点：1个padding节点 + n个有效节点
                graph.add_nodes(n+1, ntype='moving')
                graph.add_nodes(n+1, ntype='base')
                
                moving_feats = [torch.zeros(3, device=device)]  # padding节点
                moving_labels = [torch.zeros(3, device=device)]  # padding节点
                moving_exparams = [torch.zeros(3, device=device)]  # padding节点
                moving_valids = [torch.tensor([0], device=device)]  # padding节点
                
                base_feats = [torch.tensor([0.0]*6+[1.0], device=device)]  # padding节点
                base_labels = [torch.tensor([0.0]*6+[1.0], device=device)]  # padding节点
                base_valids = [torch.tensor([0], device=device)]  # padding节点
                base_multi_tags = [torch.tensor([0], device=device)]  # padding节点
                
                for i in range(n):
                    pose_pred = pose[bi, k, i]
                    pose_gt = label_pose[bi, k, i]
                    ### moving nodes ###
                    moving_feats.append(pose_pred[:3])
                    moving_labels.append(pose_gt[:3])
                    moving_exparams.append(torch.zeros(3, device=device))
                    moving_valids.append(torch.tensor([1], device=device))
                    
                    ### base nodes ###
                    base_feats.append(pose_pred)
                    base_labels.append(pose_gt)
                    base_valids.append(torch.tensor([1], device=device))
                    base_multi_tags.append(torch.tensor([0], device=device))
                
                graph.nodes['moving'].data['feat'] = torch.stack(moving_feats)
                graph.nodes['moving'].data['label'] = torch.stack(moving_labels)
                graph.nodes['moving'].data['exparam'] = torch.stack(moving_exparams)
                graph.nodes['moving'].data['valid'] = torch.stack(moving_valids).squeeze(-1)
                
                graph.nodes['base'].data['feat'] = torch.stack(base_feats)
                graph.nodes['base'].data['label'] = torch.stack(base_labels)
                graph.nodes['base'].data['valid'] = torch.stack(base_valids).squeeze(-1)
                graph.nodes['base'].data['multi_tag'] = torch.stack(base_multi_tags).squeeze(-1)
                
                ### moving2fixed edge -- others to ref ###
                src_ids = []
                dst_ids = []
                eids = []
                feats = []
                labels = []
                
                for i in range(n):
                    src_ids.append(i+1)
                    dst_ids.append(0)
                    
                    src_embed_id = ind2id(k, i)*self.max_tag_num + 0 # tag_num is 1
                    dst_embed_id = k*self.max_tag_num + 0
                    eids.append([src_embed_id, dst_embed_id])
                    
                    if self.use_uwb_seq:
                        dis = uwb_range_seq[bi, k, i]
                        feats.append(dis)
                    else:
                        dis = uwb_range_seq[bi, k, i, -1]
                        feats.append(dis.unsqueeze(0))
                    
                    gt_dis = uwb_range_gt[bi, k, i, 0]
                    labels.append(gt_dis.unsqueeze(0))
                
                if src_ids:
                    graph.add_edges(
                        src_ids, dst_ids, 
                        etype=('moving', 'moving2fixed', 'fixed'),
                        data={'eid': torch.tensor(eids, device=device), 
                              'feat': torch.stack(feats), 
                              'label': torch.stack(labels)}
                    )
                
                ### fixed2moving edge -- ref to others ###
                src_ids = []
                dst_ids = []
                eids = []
                feats = []
                labels = []
                
                for i in range(n):
                    src_ids.append(0)
                    dst_ids.append(i+1)
                    
                    id_fixed = k
                    id_moving = ind2id(k, i)
                    src_embed_id = id_fixed*self.max_tag_num + 0 # tag_num is 1
                    dst_embed_id = id_moving*self.max_tag_num + 0
                    eids.append([src_embed_id, dst_embed_id])
                    
                    if self.use_uwb_seq:
                        dis = uwb_range_seq[bi, id_moving, id2ind(id_moving, id_fixed)]
                        feats.append(dis)
                    else:
                        dis = uwb_range_seq[bi, id_moving, id2ind(id_moving, id_fixed), -1]
                        feats.append(dis.unsqueeze(0))
                    
                    gt_dis = uwb_range_gt[bi, id_moving, id2ind(id_moving, id_fixed), 0]
                    labels.append(gt_dis.unsqueeze(0))
                
                if src_ids:
                    graph.add_edges(
                        src_ids, dst_ids, 
                        etype=('fixed', 'fixed2moving', 'moving'),
                        data={'eid': torch.tensor(eids, device=device), 
                              'feat': torch.stack(feats), 
                              'label': torch.stack(labels)}
                    )
                
                ### moving2moving edge -- others to others ###
                src_ids = []
                dst_ids = []
                eids = []
                feats = []
                labels = []
                
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        
                        src_ids.append(i+1)
                        dst_ids.append(j+1)
                        
                        id_moving1 = ind2id(k, i)
                        id_moving2 = ind2id(k, j)
                        src_embed_id = id_moving1*self.max_tag_num + 0 # tag_num is 1
                        dst_embed_id = id_moving2*self.max_tag_num + 0
                        eids.append([src_embed_id, dst_embed_id])
                        
                        # i to j
                        if self.use_uwb_seq:
                            dis = uwb_range_seq[bi, id_moving2, id2ind(id_moving2, id_moving1)]
                            feats.append(dis)
                        else:
                            dis = uwb_range_seq[bi, id_moving2, id2ind(id_moving2, id_moving1), -1]
                            feats.append(dis.unsqueeze(0))
                        
                        gt_dis = uwb_range_gt[bi, id_moving2, id2ind(id_moving2, id_moving1), 0]
                        labels.append(gt_dis.unsqueeze(0))
                
                if src_ids:
                    graph.add_edges(
                        src_ids, dst_ids, 
                        etype=('moving', 'moving2moving', 'moving'),
                        data={'eid': torch.tensor(eids, device=device), 
                              'feat': torch.stack(feats), 
                              'label': torch.stack(labels)}
                    )
                
                ### moving2base edge -- others tag to others base ###
                src_ids = []
                dst_ids = []
                feats = []
                
                for i in range(n):
                    src_ids.append(i+1)
                    dst_ids.append(i+1)
                    feats.append(torch.zeros(3, device=device))
                
                if src_ids:
                    graph.add_edges(
                        src_ids, dst_ids, 
                        etype=('moving', 'moving2base', 'base'),
                        data={'feat': torch.stack(feats)}
                    )
                
                ### ref2base edge -- ref to others base ###
                src_ids = [0]
                dst_ids = [0]
                feats = [torch.zeros(3, device=device)]
                
                for i in range(n):
                    if bearing_mask[bi, k, i]:
                        src_ids.append(0)
                        dst_ids.append(i+1)
                        feats.append(bearing[bi, k, i])
                
                graph.add_edges(
                    src_ids, dst_ids, 
                    etype=('ref', 'ref2base', 'base'),
                    data={'feat': torch.stack(feats)}
                )
                
                graph_list.append(graph)
        
        batch_graph = dgl.batch(graph_list)

        return batch_graph
