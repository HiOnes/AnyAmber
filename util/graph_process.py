import torch
# import dgl.data
# from dgl.dataloading import GraphDataLoader
import dgl

def rm_node(g, rm_ntype):
    keep_ntypes = [ntype for ntype in g.ntypes if ntype not in rm_ntype]
    sub_g = dgl.node_type_subgraph(g, keep_ntypes)
    return sub_g
    # nodes_dict = {}
    # for ntype in g.ntypes:
    #     if ntype != remove_ntype:
    #         nodes_dict[ntype] = g.nodes(ntype)
    # return dgl.node_subgraph(g, nodes_dict)

def rm_prior(g, rm_ntype):
    """"
    Remove prior feats in g
    Input:
        g: dgl graph
        rm_ntype: list of node types to remove prior
    Modify:
        g: dgl graph with prior removed
    """
    for ntype in rm_ntype:
        ori_feat = g.nodes[ntype].data['feat']
        dim = ori_feat.shape[-1]
        # g.nodes[ntype].data['feat'] = torch.zeros_like(ori_feat)
        if dim == 7:
            g.nodes[ntype].data['feat'] = torch.tensor([[0,0,0,0,0,0,1]], dtype=ori_feat.dtype, device=g.device).repeat(ori_feat.shape[0],1)
        elif dim == 3:
            g.nodes[ntype].data['feat'] = torch.zeros_like(ori_feat)
        else:
            raise NotImplementedError(f'Not implemented prior removal for {ntype} with dim {dim}')

def prompted_prior(g, rm_ntype, rm_prob):
    '''
    Remove prior feats in g
    '''
    for ntype in rm_ntype:
        ori_feat = g.nodes[ntype].data['feat']
        dim = ori_feat.shape[-1]
        replace_feat = ori_feat.clone()
        bs = ori_feat.shape[0]
        indices = torch.randperm(bs, device=g.device)[:int(bs*rm_prob)]
        flag = torch.ones(bs, 1, dtype=ori_feat.dtype, device=g.device)
        flag[indices] = 0.
        if dim == 7:
            replace_feat[indices] = torch.tensor([0,0,0,0,0,0,1], dtype=ori_feat.dtype, device=g.device)
        elif dim == 3:
            replace_feat[indices] = torch.tensor([0,0,0], dtype=ori_feat.dtype, device=g.device)
        else:
            raise NotImplementedError(f'Not implemented prior removal for {ntype} with dim {dim}')
        replace_feat = torch.cat([replace_feat, flag], dim=1)
        g.nodes[ntype].data['feat'] = replace_feat

def rm_edge(g, rm_etype):
    keep_etypes = [etype for etype in g.canonical_etypes
               if etype not in rm_etype]
    sub_g = dgl.edge_type_subgraph(g, keep_etypes)
    return sub_g

def rm_edge_keep_batch(g, rm_etype):
    """
    Remove edge types from a (possibly batched) heterograph while preserving
    batch metadata.

    Args:
        g: DGLGraph, can be a batched heterograph.
        rm_etype: iterable of canonical etypes to remove.

    Returns:
        sub_g: DGLGraph after edge-type filtering, with batch metadata restored
               when available.
    """
    keep_etypes = [etype for etype in g.canonical_etypes if etype not in rm_etype]
    sub_g = dgl.edge_type_subgraph(g, keep_etypes)

    # edge_type_subgraph creates a new graph and may drop batch partitions.
    # Copy batch node/edge counts from the input graph to keep per-sample splits.
    if hasattr(g, 'batch_size') and g.batch_size is not None:
        batch_num_nodes = {}
        for ntype in sub_g.ntypes:
            try:
                batch_num_nodes[ntype] = g.batch_num_nodes(ntype)
            except Exception:
                print(f"Warning: batch_num_nodes for node type {ntype} not found in input graph. Batch metadata may be incomplete in the output graph.")
                pass
        if batch_num_nodes:
            sub_g.set_batch_num_nodes(batch_num_nodes)

        batch_num_edges = {}
        for etype in sub_g.canonical_etypes:
            try:
                batch_num_edges[etype] = g.batch_num_edges(etype)
            except Exception:
                print(f"Warning: batch_num_edges for edge type {etype} not found in input graph. Batch metadata may be incomplete in the output graph.")
                pass
        if batch_num_edges:
            sub_g.set_batch_num_edges(batch_num_edges)

    return sub_g

def add_noise_to_prior(g, ntype, t, noise):
    """
    Add noise to prior feats in g
    Input:
        g: dgl graph
        ntype: node type to add noise
        t: timesteps
        noise: noise to add
    Modify:
        g: dgl graph with prior added noise
    """
    ori_feat = g.nodes[ntype].data['feat']
    dim = ori_feat.shape[-1]
    if dim == 7:
        clean_prior = ori_feat[:, :7]
        noised_prior = clean_prior + noise * t[:, None]
        g.nodes[ntype].data['feat'] = torch.cat([noised_prior, ori_feat[:, 7:]], dim=-1)
    elif dim == 3:
        clean_prior = ori_feat[:, :3]
        noised_prior = clean_prior + noise * t[:, None]
        g.nodes[ntype].data['feat'] = torch.cat([noised_prior, ori_feat[:, 3:]], dim=-1)
    else:
        raise NotImplementedError(f'Not implemented prior noise addition for {ntype} with dim {dim}')
    
def get_retain_nodes(graph, mode):
    '''
    Get retain nodes based on flag / graph structure

    input:
        graph: DGLGraph
        mode: str
            Allowed values are 'flag', 'gs'.
    output:
        nodes_retain: dict

    '''
    nodes_retain = {}
    if mode == 'flag':
        for ntype in graph.ntypes:
            if ntype in graph.ndata['valid'].keys():
                mask = graph.ndata['valid'][ntype].bool()
            else:
                # set the mask all True
                mask = torch.ones(graph.num_nodes(ntype), dtype=torch.bool, device=graph.device)
            nodes_retain[ntype] = mask
    elif mode == 'gs':
        for ntype in graph.ntypes:
            # find the edge type whose source / target node type is ntype
            source_etypes = [etype for etype in graph.canonical_etypes if etype[0] == ntype]
            target_etypes = [etype for etype in graph.canonical_etypes if etype[2] == ntype]
            if not source_etypes and not target_etypes:
                # if there aren't out-edges / in-edges, abort the nodes
                nodes_retain[ntype] = torch.zeros(graph.num_nodes(ntype), dtype=torch.bool, device=graph.device)
                continue
            # init mask all False
            mask = torch.zeros(graph.num_nodes(ntype), dtype=torch.bool, device=graph.device)
            for etype in source_etypes:
                # calculate the out-degree of the edge
                out_deg = graph.out_degrees(etype=etype)
                mask |= out_deg > 0
            for etype in target_etypes:
                # calculate the in-degree of the edge
                in_deg = graph.in_degrees(etype=etype)
                mask |= in_deg > 0  # if any in-degree > 0, retain the nodes
            nodes_retain[ntype] = mask
    else:
        raise ValueError(f"Invalid mode: {mode}. Allowed values are 'flag' and 'gs'.")
    return nodes_retain

#################### precess data for PGO or guided diffusion ####################
def get_pgo_input(g, net_out, rf_out, nodes_retain_dict, args):
    """
    Process graph data and model outputs to get PGO inputs for single-robot single-frame setting
    Input:
        g: DGLGraph
        net_out: dict of model outputs, including 'moving_pos', 'moving_cov', 'base_pose', 'base_cov'
        rf_out: dict of range filter outputs, including 'range_cov'
        nodes_retain_dict: dict of node retain masks for each node type
        args: arguments, including batch_size, robot_num, tag_num, fix_PGO_weight
    Output:
        tag_pos: [bs, nf, n, n-1, m, 3]
        tag_cov: [bs, nf, n, n-1, m, 3]
        tag_mask: [bs, nf, n, n-1, m]
        tag_exparam: [bs, nf, n, n-1, m, 3]
        fixed_pos: [bs, nf, n, m, 3]
        fixed_mask: [bs, nf, n, m]
        dis: [bs, nf, n, n-1, m, m, 1]
        dis_cov: [bs, nf, n, n-1, m, m, 1]
        base_pose: [bs, nf, n, n-1, 7]
        base_cov: [bs, nf, n, n-1, 6]
        multi_tag_mask: [bs, nf, n, n-1]
    """
    device = g.device
    ##### tag_pos #####
    tag_pos = g.ndata['feat']['moving'].clone()[:, :3] # [batch_size*tag_num, 3]
    tag_retain = nodes_retain_dict['moving'] # [batch_size*tag_num]
    tag_pos[tag_retain, :] = net_out['moving_pos'][-1]
    tag_pos = tag_pos.reshape(-1, 1, args.tag_num, 3) # [batch_size, frame_num, tag_num, 3]
    ##### tag_cov #####
    tag_cov = torch.ones_like(g.ndata['feat']['moving'][:, :3])*1e4 # [batch_size*tag_num, 3]
    tag_cov[tag_retain, :] = net_out['moving_cov']
    tag_cov = tag_cov.reshape(-1, 1, args.tag_num, 3) # [batch_size, frame_num, tag_num, 3]
    ##### tag_mask #####
    tag_mask = ~nodes_retain_dict['moving'].reshape(-1, 1, args.tag_num) # [batch_size, frame_num, tag_num]
    ##### tag_exparam #####
    tag_exparam = g.ndata['exparam']['moving'].reshape(-1, 1, args.tag_num, 3) # [batch_size, frame_num, tag_num, 3]
    ##### anchor_pos #####
    anchor_pos = g.ndata['feat']['fixed'].reshape(-1, 1, args.anchor_num, 3) # [batch_size, frame_num, anchor_num, 3]
    ##### anchor mask #####
    anchor_mask = ~nodes_retain_dict['fixed'].reshape(-1, 1, args.anchor_num) # [batch_size, frame_num, anchor_num]
    ##### dis && dis_cov #####
    dis = torch.zeros((tag_pos.shape[0], args.tag_num, args.anchor_num), device=device) # [batch_size, tag_num, anchor_num]
    dis_cov = torch.ones_like(dis)*1e4 # [batch_size, tag_num, anchor_num]
    valid_dis_mask = torch.zeros_like(dis).bool()
    batch_num_edges = g.batch_num_edges(etype='moving2fixed')
    offsets_edges = torch.cat([torch.tensor([0], device=device), torch.cumsum(batch_num_edges, dim=0)])
    edge_ids = torch.arange(g.num_edges(etype='moving2fixed'), device=device)
    edge_batch = torch.searchsorted(offsets_edges, edge_ids, right=True) - 1
    src_id, dst_id = g.edges(etype='moving2fixed')
    src_id = src_id - edge_batch*args.tag_num
    dst_id = dst_id - edge_batch*args.anchor_num
    valid_dis_mask[edge_batch, src_id, dst_id] = True
    dis, valid_dis_mask = dis.transpose(1,2).flatten(), valid_dis_mask.transpose(1,2).flatten()
    dis_cov = dis_cov.transpose(1,2).flatten()
    valid_dis = g.edata['feat'][('moving', 'moving2fixed', 'fixed')]
    valid_dis_cov = rf_out['range_cov'][('moving', 'moving2fixed', 'fixed')]
    dis[valid_dis_mask] = valid_dis.squeeze(1)
    dis_cov[valid_dis_mask] = valid_dis_cov.squeeze(1)
    dis = dis.reshape(tag_pos.shape[0], 1, args.anchor_num, args.tag_num).transpose(2,3) # [batch_size, 1, tag_num, anchor_num]
    dis_cov = dis_cov.reshape(tag_pos.shape[0], 1, args.anchor_num, args.tag_num).transpose(2,3).unsqueeze(-1) # [batch_size, 1, tag_num, anchor_num, 1]
    ##### base_pose #####
    # uav_pose = net_out['base_pose'][-1].unsqueeze(1) # [batch_size, frame_num, 7]
    base_valid_mask = nodes_retain_dict['base'] # [batch_size*base_num]
    multi_tag = g.ndata['multi_tag']['base'].bool()[base_valid_mask]
    bp = net_out['base_pose'][-1]
    # set fixed orientation for single-robot scenario
    bp[~multi_tag, 3:7] = torch.tensor([0,0,0,1], dtype=bp.dtype, device=device).unsqueeze(0).repeat((~multi_tag).sum().item(), 1)
    base_pose = bp.unsqueeze(1) # [batch_size, frame_num, 7]
    ##### base_cov #####
    base_cov = net_out['base_cov'].unsqueeze(1) # [batch_size, frame_num, 6]
    ##### base mask #####
    base_mask = ~multi_tag.reshape(-1, 1, 1) # [batch_size*valid_base_num, 1, 1]
    # uav_mask = (torch.norm(g.ndata['label']['base'][:, 3:], p=2.0, dim=-1) < 1e-4).reshape(-1, 1, 1) # [batch_size*base_num]
    return tag_pos, tag_cov, tag_mask, tag_exparam, anchor_pos, anchor_mask, dis, dis_cov, base_pose, base_cov, base_mask

def get_pgo_input_multi(g, net_out, rf_out, nodes_retain_dict, args):
    """
    Process graph data and model outputs to get PGO inputs for multi-robot single-frame setting
    Input:
        g: DGLGraph
        net_out: dict of model outputs, including 'moving_pos', 'moving_cov', 'base_pose', 'base_cov'
        rf_out: dict of range filter outputs, including 'range_cov'
        nodes_retain_dict: dict of node retain masks for each node type
        args: arguments, including batch_size, robot_num, tag_num, fix_PGO_weight
    Output:
        tag_pos: [bs, nf, n, n-1, m, 3]
        tag_cov: [bs, nf, n, n-1, m, 3]
        tag_mask: [bs, nf, n, n-1, m]
        tag_exparam: [bs, nf, n, n-1, m, 3]
        fixed_pos: [bs, nf, n, m, 3]
        fixed_mask: [bs, nf, n, m]
        dis: [bs, nf, n, n-1, m, m, 1]
        dis_cov: [bs, nf, n, n-1, m, m, 1]
        base_pose: [bs, nf, n, n-1, 7]
        base_cov: [bs, nf, n, n-1, 6]
        multi_tag_mask: [bs, nf, n, n-1]
    """
    device = g.device
    # n is the number of robots, m is the number of tags, both not including the padding node
    bs, n, m = args.batch_size, args.robot_num, args.tag_num
    # bs = net_out['base_cov'].reshape(-1, n, n-1, 6).shape[0]
    ##### tag_pos #####
    tag_pos = net_out['moving_pos'][-1].reshape(bs, 1, n, n-1, m, 3) # [bs, nf, n, n-1, m, 3]
    ##### tag_cov #####
    tag_cov = net_out['moving_cov'].reshape(tag_pos.shape) # [bs, nf, n, n-1, m, 3]
    ##### tag_mask #####
    tag_mask = torch.zeros((bs, 1, n, n-1, m), dtype=bool, device=device) # [bs, nf, n, n-1, m]
    ##### tag_exparam #####
    tag_retain = nodes_retain_dict['moving']
    tag_exparam = g.ndata['exparam']['moving'][tag_retain, :].reshape(tag_pos.shape) # [bs, nf, n, n-1, m, 3]
    ##### fixed_pos #####
    fixed_pos = g.ndata['feat']['fixed'].reshape(bs, 1, n, m, 3) # [bs, nf, n, m, 3]
    ##### fixed mask #####
    fixed_mask = torch.zeros((bs, 1, n, m), dtype=bool, device=device) # [bs, nf, n, m]
    ##### dis && dis_cov #####
    em2f = g.edata['feat'][('moving', 'moving2fixed', 'fixed')] # [bs*n*(n-1)*m*m, 1]
    dis = em2f.reshape(bs, n, n-1, m, m, 1) # [bs, n, n-1, m, m, 1]
    em2f_cov = rf_out['range_cov'][('moving', 'moving2fixed', 'fixed')] # [bs*n*(n-1)*m*m, 1]
    dis_cov = em2f_cov.reshape(dis.shape) # [bs, n, n-1, m, m, 1]
    dis_cov[dis < 1e-4] = 1e4
    dis, dis_cov = dis.unsqueeze(1), dis_cov.unsqueeze(1) # [bs, nf, n, n-1, m, m, 1]
    ##### base_pose #####
    base_retain = nodes_retain_dict['base']
    multi_tag = g.ndata['multi_tag']['base'].bool()[base_retain]
    if m == 1:
        assert (~multi_tag).all(), "tag_num is set to 1, while part of multi_tag flag is True!"
    else:
        assert multi_tag.all(), "tag_num is not 1, while multi_tag flag is not all True!"
    base_pose = net_out['base_pose'][-1].clone()
    if not multi_tag.all():
        base_pose[:, 3:7] = g.ndata['feat']['base'][base_retain, 3:7]
    base_pose = base_pose.reshape(bs, 1, n, n-1, 7) # [bs, nf, n, n-1, 7]
    ##### base_cov #####
    base_cov = net_out['base_cov'].reshape(bs, 1, n, n-1, 6) # [bs, nf, n, n-1, 6]
    if not multi_tag.all():
        base_cov[..., 3:] = torch.ones((bs, 1, n, n-1, 3), dtype=base_cov.dtype, device=device)
        # base_cov = torch.ones_like(base_cov)
    ##### multi_tag mask #####
    multi_tag_mask = ~multi_tag.reshape(bs, 1, n, n-1) # [bs, nf, n, n-1]
    
    return tag_pos, tag_cov, tag_mask, tag_exparam, fixed_pos, fixed_mask, dis, dis_cov, base_pose, base_cov, multi_tag_mask

def get_pgo_input_multi_seq(g, net_out, rf_out, nodes_retain_dict, args):
    """
    Process graph data and model outputs to get PGO inputs for multi-robot multi-frame setting
    Input:
        g: DGLGraph
        net_out: dict of model outputs, including 'moving_pos', 'moving_cov', 'base_pose', 'base_cov'
        rf_out: dict of range filter outputs, including 'range_cov'
        nodes_retain_dict: dict of node retain masks for each node type
        args: arguments, including batch_size, robot_num, tag_num, fix_PGO_weight
    Output:
        tag_pos: [bs, nf, n, n-1, m, 3]
        tag_cov: [bs, nf, n, n-1, m, 3]
        tag_mask: [bs, nf, n, n-1, m]
        tag_exparam: [bs, nf, n, n-1, m, 3]
        fixed_pos: [bs, nf, n, m, 3]
        fixed_mask: [bs, nf, n, m]
        dis: [bs, nf, n, n-1, m, m, 1]
        dis_cov: [bs, nf, n, n-1, m, m, 1]
        base_pose: [bs, nf, n, n-1, 7]
        base_cov: [bs, nf, n, n-1, 6]
        multi_tag_mask: [bs, nf, n, n-1]
    """
    device = g.device
    # n is the number of robots, m is the number of tags, both not including the padding node
    bs, n, m = args.batch_size, args.robot_num, args.tag_num
    bs = net_out['base_cov'].reshape(-1, n, n-1, 6).shape[0]
    ##### tag_pos #####
    tag_pos = net_out['moving_pos'][-1].reshape(bs, 1, n, n-1, m, 3) # [bs, nf, n, n-1, m, 3]
    ##### tag_cov #####
    tag_cov = net_out['moving_cov'].reshape(tag_pos.shape) # [bs, nf, n, n-1, m, 3]
    ##### tag_mask #####
    tag_mask = torch.zeros((bs, 1, n, n-1, m), dtype=bool, device=device) # [bs, nf, n, n-1, m]
    ##### tag_exparam #####
    tag_retain = nodes_retain_dict['moving']
    tag_exparam = g.ndata['exparam']['moving'][tag_retain, :].reshape(tag_pos.shape) # [bs, nf, n, n-1, m, 3]
    ##### fixed_pos #####
    fixed_pos = g.ndata['feat']['fixed'].reshape(bs, 1, n, m, 3) # [bs, nf, n, m, 3]
    ##### fixed mask #####
    fixed_mask = torch.zeros((bs, 1, n, m), dtype=bool, device=device) # [bs, nf, n, m]
    ##### dis && dis_cov #####
    em2f = g.edata['feat'][('moving', 'moving2fixed', 'fixed')] # [bs*n*m*(n-1), 1]
    dis = em2f.reshape(bs, n, m, n-1).transpose(2,3) # [bs, n, n-1, m]
    dis = torch.diag_embed(dis).unsqueeze(-1) # [bs, n, n-1, m, m, 1]
    em2f_cov = rf_out['range_cov'][('moving', 'moving2fixed', 'fixed')] # [bs*n*m*(n-1), 1]
    dis_cov = em2f_cov.reshape(bs, n, m, n-1).transpose(2,3) # [bs, n, n-1, m]
    dis_cov = torch.diag_embed(dis_cov).unsqueeze(-1) # [bs, n, n-1, m, m, 1]
    dis_cov[dis < 1e-4] = 1e4
    dis, dis_cov = dis.unsqueeze(1), dis_cov.unsqueeze(1) # [bs, nf, n, n-1, m, m, 1]
    ##### base_pose #####
    base_retain = nodes_retain_dict['base']
    multi_tag = g.ndata['multi_tag']['base'].bool()[base_retain]
    if m == 1:
        assert (~multi_tag).all(), "tag_num is set to 1, while part of multi_tag flag is True!"
    else:
        assert multi_tag.all(), "tag_num is not 1, while multi_tag flag is not all True!"
    base_pose = net_out['base_pose'][-1].clone()
    if not multi_tag.all():
        base_pose[:, 3:7] = g.ndata['feat']['base'][base_retain, 3:7]
    base_pose = base_pose.reshape(bs, 1, n, n-1, 7) # [bs, nf, n, n-1, 7]
    ##### base_cov #####
    base_cov = net_out['base_cov'].reshape(bs, 1, n, n-1, 6) # [bs, nf, n, n-1, 6]
    if not multi_tag.all():
        base_cov[..., 3:] = torch.ones((bs, 1, n, n-1, 3), dtype=base_cov.dtype, device=device)
        # base_cov = torch.ones_like(base_cov)
    ##### multi_tag mask #####
    multi_tag_mask = ~multi_tag.reshape(bs, 1, n, n-1) # [bs, nf, n, n-1]
    
    return tag_pos, tag_cov, tag_mask, tag_exparam, fixed_pos, fixed_mask, dis, dis_cov, base_pose, base_cov, multi_tag_mask

##### GGS #####
def get_pgo_input_multi_seq_GGS(g, net_out, rf_out, nodes_retain_dict, args):
    """
    Process graph data and model outputs to get PGO inputs for multi-robot multi-frame setting
    Input:
        g: DGLGraph
        net_out: dict of model outputs, including 'moving_pos', 'moving_cov', 'base_pose', 'base_cov'
        rf_out: dict of range filter outputs, including 'range_cov'
        nodes_retain_dict: dict of node retain masks for each node type
        args: arguments, including batch_size, robot_num, tag_num, fix_PGO_weight
    Output:
        tag_pos: [bs, nf, n, n-1, m, 3]
        tag_cov: [bs, nf, n, n-1, m, 3]
        tag_mask: [bs, nf, n, n-1, m]
        tag_exparam: [bs, nf, n, n-1, m, 3]
        fixed_pos: [bs, nf, n, m, 3]
        fixed_mask: [bs, nf, n, m]
        dis: [bs, nf, n, n-1, m, m, 1]
        dis_cov: [bs, nf, n, n-1, m, m, 1]
        base_pose: [bs, nf, n, n-1, 7]
        base_cov: [bs, nf, n, n-1, 6]
        multi_tag_mask: [bs, nf, n, n-1]
    """
    device = g.device
    # n is the number of robots, m is the number of tags, both not including the padding node
    bs, nf, n, m = args.batch_size, args.frame_win, args.robot_num, args.tag_num
    # bs = net_out['base_cov'].reshape(-1, n, n-1, 6).shape[0]
    ##### tag_pos #####
    tag_pos = net_out['moving_pos'][-1].reshape(bs, n, n-1, nf, m, 3).permute(0, 3, 1, 2, 4, 5) # [bs, nf, n, n-1, m, 3]
    ##### tag_cov #####
    tag_cov = net_out['moving_cov'].reshape(bs, n, n-1, nf, m, 3).permute(0, 3, 1, 2, 4, 5) # [bs, nf, n, n-1, m, 3]
    ##### tag_mask #####
    tag_mask = torch.zeros((bs, nf, n, n-1, m), dtype=bool, device=device) # [bs, nf, n, n-1, m]
    ##### tag_exparam #####
    tag_retain = nodes_retain_dict['moving']
    tag_exparam = g.ndata['exparam']['moving'][tag_retain, :].reshape(bs, n, n-1, nf, m, 3).permute(0, 3, 1, 2, 4, 5) # [bs, nf, n, n-1, m, 3]
    ##### fixed_pos #####
    fixed_pos = g.ndata['feat']['fixed'].reshape(bs, n, nf, m, 3).permute(0, 2, 1, 3, 4) # [bs, nf, n, m, 3]
    ##### fixed mask #####
    fixed_mask = torch.zeros((bs, nf, n, m), dtype=bool, device=device) # [bs, nf, n, m]
    ##### dis && dis_cov #####
    em2f = g.edata['feat'][('moving', 'moving2fixed', 'fixed')] # [bs * n * nf * n-1 * m * m, 1]
    em2f_cov = rf_out['range_cov'][('moving', 'moving2fixed', 'fixed')] # [bs * n * nf * n-1 * m * m, 1]
    dis, dis_cov = [v.reshape(bs, n, nf, n-1, m, m, 1).permute(0, 2, 1, 3, 4, 5, 6) for v in [em2f, em2f_cov]] # [bs, nf, n, n-1, m, m, 1]
    dis_cov[dis < 1e-4] = 1e4

    ##### subbase_pose #####
    subbase_retain = nodes_retain_dict['subbase']
    multi_tag = g.ndata['multi_tag']['subbase'].bool()[subbase_retain]
    subbase_pose = net_out['subbase_pose'][-1].clone()
    if not multi_tag.all():
        subbase_pose[:, 3:7] = g.ndata['feat']['subbase'][subbase_retain, 3:7]
    subbase_pose = subbase_pose.reshape(bs, n, n-1, nf, 7).permute(0, 3, 1, 2, 4) # [bs, nf, n, n-1, 7]
    ##### subbase_cov #####
    subbase_cov = net_out['subbase_cov'].reshape(bs, n, n-1, nf, 6).permute(0, 3, 1, 2, 4) # [bs, nf, n, n-1, 6]
    if not multi_tag.all():
        subbase_cov[..., 3:] = torch.ones((bs, nf, n, n-1, 3), dtype=subbase_cov.dtype, device=device)
        # subbase_cov = torch.ones_like(subbase_cov)
    ##### multi_tag mask #####
    multi_tag_mask = ~multi_tag.reshape(bs, n, n-1, nf).permute(0, 3, 1, 2) # [bs, nf, n, n-1]

    # ##### base_pose #####
    # base_retain = nodes_retain_dict['base']
    # multi_tag = g.ndata['multi_tag']['base'].bool()[base_retain]
    # if m == 1:
    #     assert (~multi_tag).all(), "tag_num is set to 1, while part of multi_tag flag is True!"
    # else:
    #     assert multi_tag.all(), "tag_num is not 1, while multi_tag flag is not all True!"
    # base_pose = net_out['base_pose'][-1].clone()
    # if not multi_tag.all():
    #     base_pose[:, 3:7] = g.ndata['feat']['base'][base_retain, 3:7]
    # base_pose = base_pose.reshape(bs, 1, n, n-1, 7) # [bs, nf, n, n-1, 7]
    # ##### base_cov #####
    # base_cov = net_out['base_cov'].reshape(bs, 1, n, n-1, 6) # [bs, nf, n, n-1, 6]
    # if not multi_tag.all():
    #     base_cov[..., 3:] = torch.ones((bs, 1, n, n-1, 3), dtype=base_cov.dtype, device=device)
    #     # base_cov = torch.ones_like(base_cov)
    # ##### multi_tag mask #####
    # multi_tag_mask = ~multi_tag.reshape(bs, 1, n, n-1) # [bs, nf, n, n-1]

    ##### odom #####
    odom = g.edata['feat'][('subbase', 'subbase2subbase', 'subbase')] # [bs * n * n-1 * nf-1 * 2, 7]
    odom = odom.reshape(bs, n, n-1, nf-1, 2, 7)[..., 1, :].permute(0, 3, 1, 2, 4) # [bs, nf-1, n, n-1, 7]
    odom_mask = torch.zeros((bs, nf-1, n, n-1), dtype=bool, device=device) # [bs, nf-1, n, n-1]
    
    return tag_pos, tag_cov, tag_mask, tag_exparam, fixed_pos, fixed_mask, \
            dis, dis_cov, subbase_pose, subbase_cov, multi_tag_mask, odom, odom_mask


def get_pgo_input_multi_seq_GGS1(g, net_out, rf_out, nodes_retain_dict, args, prior_subbase_pose=None):
    """
    Process graph data and model outputs to get PGO inputs for multi-robot multi-frame setting
    Input:
        g: DGLGraph
        net_out: dict of model outputs, including 'moving_pos', 'moving_cov', 'base_pose', 'base_cov'
        rf_out: dict of range filter outputs, including 'range_cov'
        nodes_retain_dict: dict of node retain masks for each node type
        args: arguments, including batch_size, robot_num, tag_num, fix_PGO_weight
        prior_subbase_pose: optional prior subbase pose, [bs*n*(n-1)*nf, 7]
    Output:
        tag_pos: [bs*n, nf, n-1, m, 3]
        tag_cov: [bs*n, nf, n-1, m, 3]
        tag_mask: [bs*n, nf, n-1, m]
        tag_exparam: [bs*n, nf, n-1, m, 3]
        ref_pose: [bs*n, nf, 7]
        fixed_pos: [bs*n, nf, m, 3]
        fixed_mask: [bs*n, nf, m]
        dis: [bs*n, nf, n-1, m, m, 1]
        dis_cov: [bs*n, nf, n-1, m, m, 1]
        base_pose: [bs*n, nf, n-1, 7]
        base_cov: [bs*n, nf, n-1, 6]
        multi_tag_mask: [bs*n, nf, n-1]
        prior: [bs*n, nf, n-1, 7]
        odom: [bs*n, nf-1, n-1, 7]
        odom_mask: [bs*n, nf-1, n-1]
        bearing: [bs*n, nf, nc, 3]
        bearing_mask: [bs*n, nf, nc]
    """
    device = g.device
    # n is the number of robots, m is the number of tags, both not including the padding node
    bs, nf, n, m, nc = args.batch_size, args.frame_win, args.robot_num, args.tag_num, args.max_cam_num
    ##### tag_pos #####
    tag_pos = net_out['moving_pos'][-1].reshape(bs*n, n-1, nf, m, 3).transpose(1,2) # [bs*n, nf, n-1, m, 3]
    ##### tag_cov #####
    tag_cov = net_out['moving_cov'].reshape(bs*n, n-1, nf, m, 3).transpose(1,2) # [bs*n, nf, n-1, m, 3]
    ##### tag_mask #####
    tag_mask = torch.zeros((bs*n, nf, n-1, m), dtype=bool, device=device) # [bs, nf, n, n-1, m]
    ##### tag_exparam #####
    tag_retain = nodes_retain_dict['moving']
    tag_exparam = g.ndata['exparam']['moving'][tag_retain, :].reshape(bs*n, n-1, nf, m, 3).transpose(1,2) # [bs*n, nf, n-1, m, 3]
    ##### ref_pose #####
    ref_retain = nodes_retain_dict['ref']
    ref_pose = g.ndata['feat']['ref'][ref_retain].reshape(bs*n, nf, 7)
    ##### fixed_pos #####
    fixed_pos = g.ndata['feat']['fixed'].reshape(bs*n, nf, m, 3) # [bs*n, nf, m, 3]
    ##### fixed mask #####
    fixed_mask = torch.zeros((bs*n, nf, m), dtype=bool, device=device) # [bs*n, nf, m]
    ##### dis && dis_cov #####
    em2f = g.edata['feat'][('moving', 'moving2fixed', 'fixed')] # [bs * n * nf * n-1 * m * m, 1]
    em2f_cov = rf_out['range_cov'][('moving', 'moving2fixed', 'fixed')] # [bs * n * nf * n-1 * m * m, 1]
    dis, dis_cov = [v.reshape(bs*n, nf, n-1, m, m, 1) for v in [em2f, em2f_cov]] # [bs*n, nf, n-1, m, m, 1]
    dis_cov[dis < 1e-4] = 1e4

    ##### subbase_pose #####
    subbase_retain = nodes_retain_dict['subbase']
    multi_tag = g.ndata['multi_tag']['subbase'].bool()[subbase_retain]
    # subbase_pose = g.ndata['label']['subbase'][subbase_retain]
    subbase_pose = net_out['subbase_pose'][-1].clone()
    if not multi_tag.all():
        subbase_pose[:, 3:7] = g.ndata['feat']['subbase'][subbase_retain, 3:7]
    subbase_pose = subbase_pose.reshape(bs*n, n-1, nf, 7).transpose(1,2) # [bs*n, nf, n-1, 7]
    ##### subbase_cov #####
    subbase_cov = net_out['subbase_cov'].reshape(bs*n, n-1, nf, 6).transpose(1,2) # [bs*n, nf, n-1, 6]
    if not multi_tag.all():
        subbase_cov[..., 3:] = torch.ones((bs*n, nf, n-1, 3), dtype=subbase_cov.dtype, device=device)
        # subbase_cov = torch.ones_like(subbase_cov)
    ##### multi_tag mask #####
    multi_tag_mask = ~multi_tag.reshape(bs*n, n-1, nf).transpose(1,2) # [bs*n, nf, n-1]

    ##### prior_subbase_pose #####
    if prior_subbase_pose is None:
        # prior = subbase_pose.clone()
        prior = None
    else:
        prior = prior_subbase_pose.reshape(bs*n, n-1, nf, 7).transpose(1,2)

    # ##### base_pose #####
    # base_retain = nodes_retain_dict['base']
    # multi_tag = g.ndata['multi_tag']['base'].bool()[base_retain]
    # if m == 1:
    #     assert (~multi_tag).all(), "tag_num is set to 1, while part of multi_tag flag is True!"
    # else:
    #     assert multi_tag.all(), "tag_num is not 1, while multi_tag flag is not all True!"
    # base_pose = net_out['base_pose'][-1].clone()
    # if not multi_tag.all():
    #     base_pose[:, 3:7] = g.ndata['feat']['base'][base_retain, 3:7]
    # base_pose = base_pose.reshape(bs, 1, n, n-1, 7) # [bs, nf, n, n-1, 7]
    # ##### base_cov #####
    # base_cov = net_out['base_cov'].reshape(bs, 1, n, n-1, 6) # [bs, nf, n, n-1, 6]
    # if not multi_tag.all():
    #     base_cov[..., 3:] = torch.ones((bs, 1, n, n-1, 3), dtype=base_cov.dtype, device=device)
    #     # base_cov = torch.ones_like(base_cov)
    # ##### multi_tag mask #####
    # multi_tag_mask = ~multi_tag.reshape(bs, 1, n, n-1) # [bs, nf, n, n-1]

    ##### odom #####
    odom = g.edata['feat'][('subbase', 'subbase2subbase', 'subbase')] # [bs * n * n-1 * nf-1 * 2, 7]
    odom = odom.reshape(bs*n, n-1, nf-1, 2, 7)[..., 1, :].transpose(1,2) # [bs*n, nf-1, n-1, 7]
    odom_mask = torch.zeros((bs*n, nf-1, n-1), dtype=bool, device=device) # [bs*n, nf-1, n-1]

    ##### bearing #####
    bearing = g.edata['feat'][('ref','ref2subbase','subbase')].reshape(bs*n, nf, nc, n-1, 3)[..., 0, :] # [bs*n, nf, nc, 3]
    bearing_mask = torch.zeros((bs*n, nf, nc), dtype=bool, device=device) # [bs*n, nf, nc]

    ### gt_bearing ###
    bearing_id = g.ndata['label_match']['subbase'][subbase_retain] # [bs*n*(n-1)*nf, 1]
    bearing_id = bearing_id.reshape(bs*n, n-1, nf, 1).transpose(1,2).expand(-1,-1,-1,3) # [bs*n, nf, n-1, 3]
    matched_bearing = torch.gather(bearing, dim=2, index=bearing_id)
    
    return tag_pos, tag_cov, tag_mask, tag_exparam, ref_pose, fixed_pos, fixed_mask, dis, dis_cov, \
            subbase_pose, subbase_cov, multi_tag_mask, prior, odom, odom_mask, bearing, bearing_mask


def get_pgo_input_multi_seq_GGS2(g, net_out, rf_out, nodes_retain_dict, args):
    """
    Process graph data and model outputs to get PGO inputs for multi-robot multi-frame setting
    Input:
        g: DGLGraph
        net_out: dict of model outputs, including 'moving_pos', 'moving_cov', 'base_pose', 'base_cov'
        rf_out: dict of range filter outputs, including 'range_cov'
        nodes_retain_dict: dict of node retain masks for each node type
        args: arguments, including batch_size, robot_num, tag_num, fix_PGO_weight
    Output:
        tag_pos: [bs*n, nf, n-1, m, 3]
        tag_cov: [bs*n, nf, n-1, m, 3]
        tag_mask: [bs*n, nf, n-1, m]
        tag_exparam: [bs*n, nf, n-1, m, 3]
        ref_pose: [bs*n, nf, 7]
        fixed_pos: [bs*n, nf, m, 3]
        fixed_mask: [bs*n, nf, m]
        dis: [bs*n, nf, n-1, m, m, 1]
        dis_cov: [bs*n, nf, n-1, m, m, 1]
        subbase_pose: [bs*n, nf, n-1, 7]
        subbase_cov: [bs*n, nf, n-1, 6]
        multi_tag_mask: [bs*n, nf, n-1]
        subbase_exparam: [bs*n, nf, n-1, 7]
        base_pose: [bs*n, n-1, 7]
        base_cov: [bs*n, n-1, 6]
        odom: [bs*n, nf-1, n-1, 7]
        odom_mask: [bs*n, nf-1, n-1]
        bearing: [bs*n, nf, nc, 3]
        bearing_mask: [bs*n, nf, nc]
    """
    device = g.device
    # n is the number of robots, m is the number of tags, both not including the padding node
    bs, nf, n, m, nc = args.batch_size, args.frame_win, args.robot_num, args.tag_num, args.max_cam_num
    # bs = net_out['base_cov'].reshape(-1, n, n-1, 6).shape[0]
    ##### tag_pos #####
    tag_pos = net_out['moving_pos'][-1].reshape(bs*n, n-1, nf, m, 3).transpose(1,2) # [bs*n, nf, n-1, m, 3]
    ##### tag_cov #####
    tag_cov = net_out['moving_cov'].reshape(bs*n, n-1, nf, m, 3).transpose(1,2) # [bs*n, nf, n-1, m, 3]
    ##### tag_mask #####
    tag_mask = torch.zeros((bs*n, nf, n-1, m), dtype=bool, device=device) # [bs, nf, n, n-1, m]
    ##### tag_exparam #####
    tag_retain = nodes_retain_dict['moving']
    tag_exparam = g.ndata['exparam']['moving'][tag_retain, :].reshape(bs*n, n-1, nf, m, 3).transpose(1,2) # [bs*n, nf, n-1, m, 3]
    ##### ref_pose #####
    ref_retain = nodes_retain_dict['ref']
    ref_pose = g.ndata['feat']['ref'][ref_retain].reshape(bs*n, nf, 7)
    ##### fixed_pos #####
    fixed_pos = g.ndata['feat']['fixed'].reshape(bs*n, nf, m, 3) # [bs*n, nf, m, 3]
    ##### fixed mask #####
    fixed_mask = torch.zeros((bs*n, nf, m), dtype=bool, device=device) # [bs*n, nf, m]
    ##### dis && dis_cov #####
    em2f = g.edata['feat'][('moving', 'moving2fixed', 'fixed')] # [bs * n * nf * n-1 * m * m, 1]
    em2f_cov = rf_out['range_cov'][('moving', 'moving2fixed', 'fixed')] # [bs * n * nf * n-1 * m * m, 1]
    dis, dis_cov = [v.reshape(bs*n, nf, n-1, m, m, 1) for v in [em2f, em2f_cov]] # [bs*n, nf, n-1, m, m, 1]
    dis_cov[dis < 1e-4] = 1e4

    ##### subbase_pose #####
    subbase_retain = nodes_retain_dict['subbase']
    multi_tag = g.ndata['multi_tag']['subbase'].bool()[subbase_retain]
    # subbase_pose = g.ndata['label']['subbase'][subbase_retain]
    subbase_pose = net_out['subbase_pose'][-1].clone()
    if not multi_tag.all():
        subbase_pose[:, 3:7] = g.ndata['feat']['subbase'][subbase_retain, 3:7]
    subbase_pose = subbase_pose.reshape(bs*n, n-1, nf, 7).transpose(1,2) # [bs*n, nf, n-1, 7]
    ##### subbase_cov #####
    subbase_cov = net_out['subbase_cov'].reshape(bs*n, n-1, nf, 6).transpose(1,2) # [bs*n, nf, n-1, 6]
    if not multi_tag.all():
        subbase_cov[..., 3:] = torch.ones((bs*n, nf, n-1, 3), dtype=subbase_cov.dtype, device=device)
        # subbase_cov = torch.ones_like(subbase_cov)
    ##### multi_tag mask #####
    multi_tag_mask = ~multi_tag.reshape(bs*n, n-1, nf).transpose(1,2) # [bs*n, nf, n-1]
    ##### subbase_exparam #####
    subbase_exparam = g.edata['feat'][('subbase', 'subbase2base', 'base')] # [bs*n*(n-1)*nf, 7]
    subbase_exparam = subbase_exparam.reshape(bs*n, n-1, nf, 7).transpose(1,2) # [bs*n, nf, n-1, 7]

    ##### base_pose #####
    base_pose = net_out['base_pose'][-1].clone().reshape(bs*n, n-1, 7) # [bs*n, n-1, 7]
    base_cov = net_out['base_cov'].reshape(bs*n, n-1, 6) # [bs*n, n-1, 6]

    ##### odom #####
    odom = g.edata['feat'][('subbase', 'subbase2subbase', 'subbase')] # [bs * n * n-1 * nf-1 * 2, 7]
    odom = odom.reshape(bs*n, n-1, nf-1, 2, 7)[..., 1, :].transpose(1,2) # [bs*n, nf-1, n-1, 7]
    odom_mask = torch.zeros((bs*n, nf-1, n-1), dtype=bool, device=device) # [bs*n, nf-1, n-1]

    ##### bearing #####
    bearing = g.edata['feat'][('ref','ref2subbase','subbase')].reshape(bs*n, nf, nc, n-1, 3)[..., 0, :] # [bs*n, nf, nc, 3]
    bearing_mask = torch.zeros((bs*n, nf, nc), dtype=bool, device=device) # [bs*n, nf, nc]

    ### gt_bearing ###
    bearing_id = g.ndata['label_match']['subbase'][subbase_retain] # [bs*n*(n-1)*nf, 1]
    bearing_id = bearing_id.reshape(bs*n, n-1, nf, 1).transpose(1,2).expand(-1,-1,-1,3) # [bs*n, nf, n-1, 3]
    matched_bearing = torch.gather(bearing, dim=2, index=bearing_id)
    
    return tag_pos, tag_cov, tag_mask, tag_exparam, ref_pose, fixed_pos, fixed_mask, dis, dis_cov, \
            subbase_pose, subbase_cov, multi_tag_mask, subbase_exparam, base_pose, base_cov, \
            odom, odom_mask, bearing, bearing_mask

def get_pgo_input_multi_seq_GGS1_recurrent(g, net_out, rf_out, nodes_retain_dict, args, prior_subbase_pose=None):
    """
    Process graph data and model outputs to get PGO inputs for multi-robot multi-frame setting
    Input:
        g: DGLGraph
        net_out: dict of model outputs, including 'moving_pos', 'moving_cov', 'base_pose', 'base_cov'
        rf_out: dict of range filter outputs, including 'range_cov'
        nodes_retain_dict: dict of node retain masks for each node type
        args: arguments, including batch_size, robot_num, tag_num, fix_PGO_weight
        prior_subbase_pose: optional prior subbase pose, [bs*n*(n-1)*nf, 7]
    Output:
        tag_pos: [bs*n, nf, n-1, m, 3]
        tag_cov: [bs*n, nf, n-1, m, 3]
        tag_mask: [bs*n, nf, n-1, m]
        tag_exparam: [bs*n, nf, n-1, m, 3]
        ref_pose: [bs*n, nf, 7]
        fixed_pos: [bs*n, nf, m, 3]
        fixed_mask: [bs*n, nf, m]
        dis: [bs*n, nf, n-1, m, m, 1]
        dis_cov: [bs*n, nf, n-1, m, m, 1]
        base_pose: [bs*n, nf, n-1, 7]
        base_cov: [bs*n, nf, n-1, 6]
        multi_tag_mask: [bs*n, nf, n-1]
        prior: [bs*n, nf, n-1, 7]
        odom: None
        odom_mask: None
        bearing: [bs*n, nf, nc, 3]
        bearing_mask: [bs*n, nf, nc]
    """
    device = g.device
    # n is the number of robots, m is the number of tags, both not including the padding node
    bs, nf, n, m, nc = args.batch_size, 1, args.robot_num, args.tag_num, args.max_cam_num
    ##### tag_pos #####
    tag_pos = net_out['moving_pos'][-1].reshape(bs*n, n-1, nf, m, 3).transpose(1,2) # [bs*n, nf, n-1, m, 3]
    ##### tag_cov #####
    tag_cov = net_out['moving_cov'].reshape(bs*n, n-1, nf, m, 3).transpose(1,2) # [bs*n, nf, n-1, m, 3]
    ##### tag_mask #####
    tag_mask = torch.zeros((bs*n, nf, n-1, m), dtype=bool, device=device) # [bs, nf, n, n-1, m]
    ##### tag_exparam #####
    tag_retain = nodes_retain_dict['moving']
    tag_exparam = g.ndata['exparam']['moving'][tag_retain, :].reshape(bs*n, n-1, nf, m, 3).transpose(1,2) # [bs*n, nf, n-1, m, 3]
    ##### ref_pose #####
    ref_retain = nodes_retain_dict['ref']
    ref_pose = g.ndata['feat']['ref'][ref_retain].reshape(bs*n, nf, 7)
    ##### fixed_pos #####
    fixed_pos = g.ndata['feat']['fixed'].reshape(bs*n, nf, m, 3) # [bs*n, nf, m, 3]
    ##### fixed mask #####
    fixed_mask = torch.zeros((bs*n, nf, m), dtype=bool, device=device) # [bs*n, nf, m]
    ##### dis && dis_cov #####
    em2f = g.edata['feat'][('moving', 'moving2fixed', 'fixed')] # [bs * n * nf * n-1 * m * m, 1]
    # em2f_cov = rf_out['range_cov'][('moving', 'moving2fixed', 'fixed')] # [bs * n * nf * n-1 * m * m, 1]
    em2f_cov = torch.ones_like(em2f) # To Do: use gru range_cov in recurrent setting
    dis, dis_cov = [v.reshape(bs*n, nf, n-1, m, m, 1) for v in [em2f, em2f_cov]] # [bs*n, nf, n-1, m, m, 1]
    dis_cov[dis < 1e-4] = 1e4

    ##### base_pose #####
    base_retain = nodes_retain_dict['base']
    multi_tag = g.ndata['multi_tag']['base'].bool()[base_retain]
    multi_tag = torch.ones_like(multi_tag, dtype=bool) # change "multi_tag" to True
    # base_pose = g.ndata['label']['base'][base_retain]
    base_pose = net_out['base_pose'][-1].clone()
    if not multi_tag.all():
        base_pose[:, 3:7] = g.ndata['feat']['base'][base_retain, 3:7]
    base_pose = base_pose.reshape(bs*n, n-1, nf, 7).transpose(1,2) # [bs*n, nf, n-1, 7]
    ##### base_cov #####
    base_cov = net_out['base_cov'].reshape(bs*n, n-1, nf, 6).transpose(1,2) # [bs*n, nf, n-1, 6]
    if not multi_tag.all():
        base_cov[..., 3:] = torch.ones((bs*n, nf, n-1, 3), dtype=base_cov.dtype, device=device)
        # base_cov = torch.ones_like(base_cov)
    ##### multi_tag mask #####
    multi_tag_mask = ~multi_tag.reshape(bs*n, n-1, nf).transpose(1,2) # [bs*n, nf, n-1]

    ##### prior_base_pose #####
    if prior_subbase_pose is None:
        prior = None
        # prior = base_pose.clone()
    else:
        prior = prior_subbase_pose.reshape(bs*n, n-1, nf, 7).transpose(1,2)

    ##### odom #####
    odom, odom_mask = None, None

    ##### bearing #####
    bearing = g.edata['feat'][('ref','ref2base','base')].reshape(bs*n, nf, nc, n-1, 3)[..., 0, :] # [bs*n, nf, nc, 3]
    bearing_mask = torch.zeros((bs*n, nf, nc), dtype=bool, device=device) # [bs*n, nf, nc]

    ### gt_bearing ###
    bearing_id = g.ndata['label_match']['base'][base_retain] # [bs*n*(n-1)*nf, 1]
    bearing_id = bearing_id.reshape(bs*n, n-1, nf, 1).transpose(1,2).expand(-1,-1,-1,3) # [bs*n, nf, n-1, 3]
    matched_bearing = torch.gather(bearing, dim=2, index=bearing_id)
    
    return tag_pos, tag_cov, tag_mask, tag_exparam, ref_pose, fixed_pos, fixed_mask, dis, dis_cov, \
            base_pose, base_cov, multi_tag_mask, prior, odom, odom_mask, bearing, bearing_mask


def get_pgo_input_multi_seq_GGS1_recurrent_list(g_list, net_out_list, rf_out, nodes_retain_dict, args, prior_subbase_pose=None):
    """
    Process graph data and model outputs to get PGO inputs for multi-robot multi-frame setting
    Input:
        g: DGLGraph
        net_out: dict of model outputs, including 'moving_pos', 'moving_cov', 'base_pose', 'base_cov'
        rf_out: dict of range filter outputs, including 'range_cov'
        nodes_retain_dict: dict of node retain masks for each node type
        args: arguments, including batch_size, robot_num, tag_num, fix_PGO_weight
        prior_subbase_pose: optional prior subbase pose, [bs*n*(n-1)*nf, 7]
    Output:
        tag_pos: [bs*n, nf, n-1, m, 3]
        tag_cov: [bs*n, nf, n-1, m, 3]
        tag_mask: [bs*n, nf, n-1, m]
        tag_exparam: [bs*n, nf, n-1, m, 3]
        ref_pose: [bs*n, nf, 7]
        fixed_pos: [bs*n, nf, m, 3]
        fixed_mask: [bs*n, nf, m]
        dis: [bs*n, nf, n-1, m, m, 1]
        dis_cov: [bs*n, nf, n-1, m, m, 1]
        base_pose: [bs*n, nf, n-1, 7]
        base_cov: [bs*n, nf, n-1, 6]
        multi_tag_mask: [bs*n, nf, n-1]
        prior: [bs*n, nf, n-1, 7]
        odom: [bs*n, nf-1, n, 7]
        odom_mask: [bs*n, nf-1, n]
        bearing: [bs*n, nf, nc, 3]
        bearing_mask: [bs*n, nf, nc]
    """
    tag_pos_list, tag_cov_list, tag_mask_list, tag_exparam_list, ref_pose_list, fixed_pos_list, fixed_mask_list, \
    dis_list, dis_cov_list, base_pose_list, base_cov_list, multi_tag_mask_list, prior_list, odom_list, odom_mask_list, bearing_list, bearing_mask_list = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for i, (g, net_out) in enumerate(zip(g_list, net_out_list)):
        device = g.device
        # n is the number of robots, m is the number of tags, both not including the padding node
        bs, nf, n, m, nc = args.batch_size, 1, args.robot_num, args.tag_num, args.max_cam_num
        ##### tag_pos #####
        tag_pos = net_out['moving_pos'][-1].reshape(bs*n, n-1, nf, m, 3).transpose(1,2) # [bs*n, nf, n-1, m, 3]
        ##### tag_cov #####
        tag_cov = net_out['moving_cov'].reshape(bs*n, n-1, nf, m, 3).transpose(1,2) # [bs*n, nf, n-1, m, 3]
        ##### tag_mask #####
        tag_mask = torch.zeros((bs*n, nf, n-1, m), dtype=bool, device=device) # [bs, nf, n, n-1, m]
        ##### tag_exparam #####
        tag_retain = nodes_retain_dict['moving']
        tag_exparam = g.ndata['exparam']['moving'][tag_retain, :].reshape(bs*n, n-1, nf, m, 3).transpose(1,2) # [bs*n, nf, n-1, m, 3]
        ##### ref_pose #####
        ref_retain = nodes_retain_dict['ref']
        ref_pose = g.ndata['feat']['ref'][ref_retain].reshape(bs*n, nf, 7)
        ##### fixed_pos #####
        fixed_pos = g.ndata['feat']['fixed'].reshape(bs*n, nf, m, 3) # [bs*n, nf, m, 3]
        ##### fixed mask #####
        fixed_mask = torch.zeros((bs*n, nf, m), dtype=bool, device=device) # [bs*n, nf, m]
        ##### dis && dis_cov #####
        em2f = g.edata['feat'][('moving', 'moving2fixed', 'fixed')] # [bs * n * nf * n-1 * m * m, 1]
        # em2f_cov = rf_out['range_cov'][('moving', 'moving2fixed', 'fixed')] # [bs * n * nf * n-1 * m * m, 1]
        em2f_cov = torch.ones_like(em2f) # To Do: use gru range_cov in recurrent setting
        dis, dis_cov = [v.reshape(bs*n, nf, n-1, m, m, 1) for v in [em2f, em2f_cov]] # [bs*n, nf, n-1, m, m, 1]
        dis_cov[dis < 1e-4] = 1e4

        ##### base_pose #####
        base_retain = nodes_retain_dict['base']
        multi_tag = g.ndata['multi_tag']['base'].bool()[base_retain]
        multi_tag = torch.ones_like(multi_tag, dtype=bool) # change "multi_tag" to True
        # base_pose = g.ndata['label']['base'][base_retain]
        base_pose = net_out['base_pose'][-1].clone()
        if not multi_tag.all():
            base_pose[:, 3:7] = g.ndata['feat']['base'][base_retain, 3:7]
        base_pose = base_pose.reshape(bs*n, n-1, nf, 7).transpose(1,2) # [bs*n, nf, n-1, 7]
        ##### base_cov #####
        base_cov = net_out['base_cov'].reshape(bs*n, n-1, nf, 6).transpose(1,2) # [bs*n, nf, n-1, 6]
        if not multi_tag.all():
            base_cov[..., 3:] = torch.ones((bs*n, nf, n-1, 3), dtype=base_cov.dtype, device=device)
            # base_cov = torch.ones_like(base_cov)
        ##### multi_tag mask #####
        multi_tag_mask = ~multi_tag.reshape(bs*n, n-1, nf).transpose(1,2) # [bs*n, nf, n-1]

        ##### prior_base_pose #####
        if prior_subbase_pose is None:
            prior = None
            # prior = base_pose.clone()
        else:
            prior = prior_subbase_pose.reshape(bs*n, n-1, nf, 7).transpose(1,2)

        ##### odom #####
        # odom, odom_mask = None, None
        ref_odom = g.ndata["odom"]["ref"][nodes_retain_dict["ref"]].reshape(-1, 1, 7) # [bs*n, 1, 7]
        base_odom = g.ndata["odom"]["base"][nodes_retain_dict["base"]].reshape(-1, n-1, 7) # [bs*n, n-1, 7]
        odom = torch.cat([ref_odom, base_odom], dim=1).unsqueeze(1) # [bs*n, nf, n, 7]
        odom_mask = torch.zeros((bs*n, nf, n), dtype=bool, device=device) # [bs*n, nf, n]

        ##### bearing #####
        bearing = g.edata['feat'][('ref','ref2base','base')].reshape(bs*n, nf, nc, n-1, 3)[..., 0, :] # [bs*n, nf, nc, 3]
        bearing_mask = torch.zeros((bs*n, nf, nc), dtype=bool, device=device) # [bs*n, nf, nc]

        ### gt_bearing ###
        bearing_id = g.ndata['label_match']['base'][base_retain] # [bs*n*(n-1)*nf, 1]
        bearing_id = bearing_id.reshape(bs*n, n-1, nf, 1).transpose(1,2).expand(-1,-1,-1,3) # [bs*n, nf, n-1, 3]
        matched_bearing = torch.gather(bearing, dim=2, index=bearing_id)

        tag_pos_list.append(tag_pos)
        tag_cov_list.append(tag_cov)
        tag_mask_list.append(tag_mask)
        tag_exparam_list.append(tag_exparam)
        ref_pose_list.append(ref_pose)
        fixed_pos_list.append(fixed_pos)
        fixed_mask_list.append(fixed_mask)
        dis_list.append(dis)
        dis_cov_list.append(dis_cov)
        base_pose_list.append(base_pose)
        base_cov_list.append(base_cov)
        multi_tag_mask_list.append(multi_tag_mask)
        prior_list.append(prior)
        if i > 0: # only use odom from the 2nd sequence onward
            odom_list.append(odom)
            odom_mask_list.append(odom_mask)
        bearing_list.append(bearing)
        bearing_mask_list.append(bearing_mask)
    
    # concatenate the lists along the frame dimension (dim=1)
    tag_pos, tag_cov, tag_mask, tag_exparam, ref_pose, fixed_pos, fixed_mask, dis, dis_cov, \
    base_pose, base_cov, multi_tag_mask, odom, odom_mask, bearing, bearing_mask = \
        [torch.cat(v_list, dim=1) for v_list in [tag_pos_list, tag_cov_list, tag_mask_list, tag_exparam_list, ref_pose_list, fixed_pos_list, fixed_mask_list, dis_list, dis_cov_list, base_pose_list, base_cov_list, multi_tag_mask_list, odom_list, odom_mask_list, bearing_list, bearing_mask_list]]

    return tag_pos, tag_cov, tag_mask, tag_exparam, ref_pose, fixed_pos, fixed_mask, dis, dis_cov, \
            base_pose, base_cov, multi_tag_mask, prior, odom, odom_mask, bearing, bearing_mask

def get_pgo_input_multi_seq_GGS1_recurrent_guide_window(g, net_out, rf_out, nodes_retain_dict, args, prior_subbase_pose=None):
    """
    Process graph data and model outputs to get PGO inputs for multi-robot multi-frame setting
    Input:
        g: DGLGraph
        net_out: dict of model outputs, including 'moving_pos', 'moving_cov', 'base_pose', 'base_cov'
        rf_out: dict of range filter outputs, including 'range_cov'
        nodes_retain_dict: dict of node retain masks for each node type
        args: arguments, including batch_size, robot_num, tag_num, fix_PGO_weight
        prior_subbase_pose: optional prior subbase pose, [bs*n*(n-1)*nf, 7]
    Output:
        tag_pos: [bs*n, nf, n-1, m, 3]
        tag_cov: [bs*n, nf, n-1, m, 3]
        tag_mask: [bs*n, nf, n-1, m]
        tag_exparam: [bs*n, nf, n-1, m, 3]
        ref_pose: [bs*n, nf, 7]
        fixed_pos: [bs*n, nf, m, 3]
        fixed_mask: [bs*n, nf, m]
        dis: [bs*n, nf, n-1, m, m, 1]
        dis_cov: [bs*n, nf, n-1, m, m, 1]
        base_pose: [bs*n, nf, n-1, 7]
        base_cov: [bs*n, nf, n-1, 6]
        multi_tag_mask: [bs*n, nf, n-1]
        prior: [bs*n, nf, n-1, 7]
        odom: None
        odom_mask: None
        bearing: [bs*n, nf, nc, 3]
        bearing_mask: [bs*n, nf, nc]
    """
    device = g.device
    # n is the number of robots, m is the number of tags, both not including the padding node
    bs, nf, n, m, nc = args.batch_size, args.guide_win, args.robot_num, args.tag_num, args.max_cam_num
    ##### tag_pos #####
    tag_pos = net_out['moving_pos'][-1].reshape(nf, bs*n, n-1, m, 3).transpose(0,1) # [bs*n, nf, n-1, m, 3]
    ##### tag_cov #####
    tag_cov = net_out['moving_cov'].reshape(nf, bs*n, n-1, m, 3).transpose(0,1) # [bs*n, nf, n-1, m, 3]
    ##### tag_mask #####
    tag_mask = torch.zeros((bs*n, nf, n-1, m), dtype=bool, device=device) # [bs, nf, n, n-1, m]
    ##### tag_exparam #####
    tag_retain = nodes_retain_dict['moving']
    tag_exparam = g.ndata['exparam']['moving'][tag_retain].reshape(nf, bs*n, n-1, m, 3).transpose(0,1) # [bs*n, nf, n-1, m, 3]
    ##### ref_pose #####
    ref_retain = nodes_retain_dict['ref']
    ref_pose = g.ndata['feat']['ref'][ref_retain].reshape(nf, bs*n, 7).transpose(0,1) # [bs*n, nf, 7]
    ##### fixed_pos #####
    fixed_pos = g.ndata['feat']['fixed'].reshape(nf, bs*n, m, 3).transpose(0,1) # [bs*n, nf, m, 3]
    ##### fixed mask #####
    fixed_mask = torch.zeros((bs*n, nf, m), dtype=bool, device=device) # [bs*n, nf, m]
    ##### dis && dis_cov #####
    em2f = g.edata['feat'][('moving', 'moving2fixed', 'fixed')] # [nf * bs * n * n-1 * m * m, 1]
    em2f_cov = torch.ones_like(em2f) # TODO: use gru range_cov in recurrent setting
    dis, dis_cov = [v.reshape(nf, bs*n, n-1, m, m, 1).transpose(0,1) for v in [em2f, em2f_cov]] # [bs*n, nf, n-1, m, m, 1]
    dis_cov[dis < 1e-4] = 1e4

    ##### base_pose #####
    base_retain = nodes_retain_dict['base']
    multi_tag = g.ndata['multi_tag']['base'].bool()[base_retain]
    multi_tag = torch.ones_like(multi_tag, dtype=bool) # change "multi_tag" to True
    # base_pose = g.ndata['label']['base'][base_retain]
    base_pose = net_out['base_pose'][-1].clone()
    if not multi_tag.all():
        base_pose[:, 3:7] = g.ndata['feat']['base'][base_retain, 3:7]
    base_pose = base_pose.reshape(nf, bs*n, n-1, 7).transpose(0,1) # [bs*n, nf, n-1, 7]
    ##### base_cov #####
    base_cov = net_out['base_cov'].reshape(nf, bs*n, n-1, 6).transpose(0,1) # [bs*n, nf, n-1, 6]
    if not multi_tag.all():
        base_cov[..., 3:] = torch.ones((bs*n, nf, n-1, 3), dtype=base_cov.dtype, device=device)
        # base_cov = torch.ones_like(base_cov)
    ##### multi_tag mask #####
    multi_tag_mask = ~multi_tag.reshape(nf, bs*n, n-1).transpose(0,1) # [bs*n, nf, n-1]

    ##### prior_base_pose #####
    if prior_subbase_pose is None:
        prior = None
        # prior = base_pose.clone()
    else:
        prior = prior_subbase_pose.reshape(nf, bs*n, n-1, 7).transpose(0,1)

    ##### odom #####
    # odom, odom_mask = None, None
    ref_odom = g.ndata["odom"]["ref"][nodes_retain_dict["ref"]].reshape(nf, bs*n, 1, 7)[1:].transpose(0,1) # [bs*n, nf-1, 1, 7]
    base_odom = g.ndata["odom"]["base"][nodes_retain_dict["base"]].reshape(nf, bs*n, n-1, 7)[1:].transpose(0,1) # [bs*n, nf-1, n-1, 7]
    odom = torch.cat([ref_odom, base_odom], dim=2) # [bs*n, nf-1, n, 7]
    odom_mask = torch.zeros((bs*n, nf-1, n), dtype=bool, device=device) # [bs*n, nf-1, n]

    ##### bearing #####
    bearing = g.edata['feat'][('ref','ref2base','base')].reshape(nf, bs*n, nc, n-1, 3)[..., 0, :].transpose(0,1) # [bs*n, nf, nc, 3]
    bearing_mask = torch.zeros((bs*n, nf, nc), dtype=bool, device=device) # [bs*n, nf, nc]

    ### gt_bearing ###
    bearing_id = g.ndata['label_match']['base'][base_retain] # [nf*bs*n*(n-1), 1]
    bearing_id = bearing_id.reshape(nf, bs*n, n-1, 1).transpose(0,1).expand(-1,-1,-1,3) # [bs*n, nf, n-1, 3]
    matched_bearing = torch.gather(bearing, dim=2, index=bearing_id)
    
    return tag_pos, tag_cov, tag_mask, tag_exparam, ref_pose, fixed_pos, fixed_mask, dis, dis_cov, \
            base_pose, base_cov, multi_tag_mask, prior, odom, odom_mask, bearing, bearing_mask
