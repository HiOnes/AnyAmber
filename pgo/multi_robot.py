import torch
import theseus as th
import util.utils as utils
import torch.nn.functional as F

def generate_theseus_input(base_opt_pose):
    '''
    input:
        base_opt_pose: torch.tensor, [bs, n-1, 7] base
    '''
    n = base_opt_pose.shape[1]+1
    base_opt_pose_clone = base_opt_pose.clone()
    inputs = {}
    for i in range(n-1):
        inputs[f"opt_base_pose_{i}"] = th.SE3(base_opt_pose_clone[:, i, :])
    return inputs

def self_dis_err_func(optim_vars, aux_vars):
    [pose_i] = optim_vars
    exp_i, node_j, dis = aux_vars
    node_i = pose_i.rotation().rotate(exp_i) + pose_i.translation()
    pred_dis = torch.norm((node_i - node_j).tensor, p=2.0, dim=-1, keepdim=True)
    err = pred_dis - dis.tensor
    return err

def cross_dis_err_func(optim_vars, aux_vars):
    pose_i, pose_j = optim_vars
    exp_i, exp_j, dis = aux_vars
    node_i = pose_i.rotation().rotate(exp_i) + pose_i.translation()
    node_j = pose_j.rotation().rotate(exp_j) + pose_j.translation()
    pred_dis = torch.norm((node_i - node_j).tensor, p=2.0, dim=-1, keepdim=True)
    err = pred_dis - dis.tensor
    return err

def exparam_err_func(optim_vars, aux_vars):
    t_node, T_base = optim_vars
    [t_ex] = aux_vars
    err = T_base.rotation().rotate(t_ex) + T_base.translation() - t_node
    return err.tensor

def cross_rel_pos_err_func(optim_vars, aux_vars):
    node_i, node_j = optim_vars # [bs, se3]
    [edge_ij] = aux_vars # [bs, se3]
    err = edge_ij.translation() - node_i.rotation().inverse().rotate(node_j.translation() - node_i.translation())   # [bs, 3]
    return err.tensor

def self_rel_pos_err_func(optim_vars, aux_vars):
    [node_i] = optim_vars # [bs, se3]
    [edge_ij] = aux_vars # [bs, se3]
    err = edge_ij.translation() - node_i.rotation().inverse().rotate(-node_i.translation())   # [bs, 3]
    return err.tensor

def get_opt_info(objective, n, m, node_opt_mask, node_aux_mask, base_opt_mask, device='cuda:0'):
    '''
    node_opt_mask: torch.tensor, [bs, n]
    node_aux_mask: torch.tensor, [bs, m]
    base_opt_mask: torch.tensor, [bs, 1]
    '''
    lin = th.DenseLinearization(objective, vectorize=True)
    lin.linearize()

    # 1. get covariance
    hessian = lin.hessian_approx()
    reg = torch.eye(hessian.size(1), device=device).unsqueeze(0) * 1e-6 #regularization term
    hessian_reg = hessian + reg
    try:
        cov_mat = torch.linalg.inv(hessian_reg)
    except:
        cov_mat = torch.linalg.pinv(hessian_reg)
    
    ### get variable indices for covariance extraction ###
    var_indices = {}
    start_col = 0
    for var in objective.optim_vars.values():
        var_dim = var.dof()
        var_indices[var.name] = (start_col, start_col + var_dim)
        start_col += var_dim
    
    ### get covariance for base pose ###
    bp_start, bp_end = var_indices["opt_base_pose"]
    base_pose_cov_mat = cov_mat[:, bp_start:bp_end, bp_start:bp_end] # [bs, 6, 6]
    base_pose_cov = base_pose_cov_mat.diagonal(offset=0, dim1=1, dim2=2) # [bs, 6]
    
    ### get covariance for node positions ###
    nodes_cov_list = []
    for i in range(n):
        start, end = var_indices[f"opt_node_pos_{i}"]
        nodes_cov_mat = cov_mat[:, start:end, start:end]
        nodes_cov_list.append(nodes_cov_mat.diagonal(offset=0, dim1=1, dim2=2)) # [bs, 3]
    nodes_cov = torch.stack(nodes_cov_list, dim=1) # [bs, n, 3]

    # 2. get opt loss
    def get_masked_loss(err, ms):
        assert err.shape == ms.shape # [bs, r]
        valid_ms = ~ms # [bs, r]
        ms_err = err * valid_ms # [bs, r]
        sum_per_row = ms_err.sum(dim=1, keepdim=True) # [bs, 1]
        cnt_per_row = valid_ms.sum(dim=1, keepdim=True).float() # [bs, 1]
        avg_per_row = sum_per_row / cnt_per_row.clamp(min=1e-8) # [bs, 1]
        loss = torch.where(cnt_per_row > 0, avg_per_row, torch.tensor(1e4, dtype=avg_per_row.dtype, device=avg_per_row.device))
        return loss
    err = lin.b
    ### get loss for node positions ###
    nodes_loss_list = []
    for i in range(n):
        err_node = err[:, i*m : (i+1)*m]**2 # [bs, m]
        ms_node = node_opt_mask[:, i].unsqueeze(-1).repeat(1,m) | node_aux_mask # [bs, m]
        node_loss = get_masked_loss(err_node, ms_node)
        nodes_loss_list.append(node_loss)
    nodes_loss = torch.stack(nodes_loss_list, dim=1) # [bs, n, 1]
    ### get loss for base pose ###
    ms_base = None
    for i in range(n):
        ms = node_opt_mask[:, i].unsqueeze(-1).repeat(1,3)
        ms_base = torch.cat((ms_base, ms), dim=-1) if ms_base is not None else ms
    ms_base = ms_base | base_opt_mask.repeat(1, 3*n) # [bs, 3*n]
    err_base = err[:, n*m : n*m+3*n]**2 # [bs, 3*n]
    base_loss = get_masked_loss(err_base, ms_base) # [bs, 1]

    return torch.cat((nodes_cov, nodes_loss), dim=-1), torch.cat((base_pose_cov, base_loss), dim=-1)

def build_pgo(ref_id, node_opt_pos, node_opt_cov, node_opt_mask, node_exparam, node_aux_pos, node_aux_mask, dis, dis_cov, base_pose_all, base_cov_all, base_mask_all, fix_weight, get_pgo_info, device='cuda:0'):
    '''
    input:
        ref_id: int, reference frame id
        node_opt_pos: torch.tensor, [bs, n-1, m, 3] tag
        node_opt_cov: torch.tensor, [bs, n-1, m, 3]
        node_opt_mask: torch.tensor, [bs, n-1, m]
        node_exparam: torch.tensor, [bs, n-1, m, 3]
        node_aux_pos: torch.tensor, [bs, m, 3] reference tag
        node_aux_mask: torch.tensor, [bs, m]
        dis: torch.tensor, [bs, n, n-1, m, m, 1] distance between tags
        dis_cov: torch.tensor, [bs, n, n-1, m, m, 1]
        base_pose_all: torch.tensor, [bs, n, n-1, 7] all base poses
        base_cov_all: torch.tensor, [bs, n, n-1, 6]
        base_mask_all: torch.tensor, [bs, n, n-1]
        device: str, device id
    output:
        out_node_pos: torch.tensor, [bs, n-1, m, 3]
        out_base_pose: torch.tensor, [bs, n-1, 7]
        pgo_info: Optional, dict containing node and base pose optimization info
    '''
    bs, n, m = node_opt_pos.size(0), node_opt_pos.size(1)+1, node_opt_pos.size(2)
    objective = th.Objective().to(device)

    #### Optimization Variables ####
    base_opted_pose = { i : th.SE3(base_pose_all[:, ref_id, i, :], name=f"opt_base_pose_{i}") for i in range(n-1)}
    
    #### weights ####
    w_tag, w_dis, w_base = [torch.sqrt(1.0/(cov*10.0) + 1e-6) for cov in (node_opt_cov, dis_cov, base_cov_all)]

    ################# dis pair cost ###############
    for i in range(n):
        i_id = i
        i_ind = utils.id2ind(ref_id=ref_id, this_id=i_id)
        for j in range(n-1):
            j_id = utils.ind2id(ref_id=i, this_ind=j)
            j_ind = utils.id2ind(ref_id=ref_id, this_id=j_id)
            ###### self dis cost ######
            if i_id == ref_id:
                for k in range(m):
                    # optim_vars = [node_opted_pos[j_ind*m + k]]
                    optim_vars = [base_opted_pose[j_ind]]
                    for l in range(m):
                        aux_vars = [th.Point3(node_exparam[:, j_ind, k, :]), th.Point3(node_aux_pos[:, l, :]), th.Variable(dis[:, i, j, k, l, :])]
                        w = th.ScaleCostWeight(w_dis[:, i, j, k, l, :])
                        objective.add(th.AutoDiffCostFunction(optim_vars=optim_vars, err_fn=self_dis_err_func, dim=1, aux_vars=aux_vars, name=f"dis_cost_{i}_{j}_{k}_{l}", cost_weight=w))
            
            elif j_id == ref_id:
                for k in range(m):
                    for l in range(m):
                        # optim_vars = [node_opted_pos[i_ind*m + l]]
                        optim_vars = [base_opted_pose[i_ind]]
                        aux_vars = [th.Point3(node_exparam[:, i_ind, l, :]), th.Point3(node_aux_pos[:, k, :]), th.Variable(dis[:, i, j, k, l, :])]
                        w = th.ScaleCostWeight(w_dis[:, i, j, k, l, :])
                        objective.add(th.AutoDiffCostFunction(optim_vars=optim_vars, err_fn=self_dis_err_func, dim=1, aux_vars=aux_vars, name=f"dis_cost_{i}_{j}_{k}_{l}", cost_weight=w))
            
            ###### cross dis cost ######
            else:
                for k in range(m):
                    for l in range(m):
                        # optim_vars = [node_opted_pos[i_ind*m + l], node_opted_pos[j_ind*m + k]]
                        optim_vars = [base_opted_pose[i_ind], base_opted_pose[j_ind]]
                        aux_vars = [th.Point3(node_exparam[:, i_ind, l, :]), th.Point3(node_exparam[:, j_ind, k, :]), th.Variable(dis[:, i, j, k, l, :])]
                        w = th.ScaleCostWeight(w_dis[:, i, j, k, l, :])
                        objective.add(th.AutoDiffCostFunction(optim_vars=optim_vars, err_fn=cross_dis_err_func, dim=1, aux_vars=aux_vars, name=f"dis_cost_{i}_{j}_{k}_{l}", cost_weight=w))

    ################# rel pos cost ###############
    for i in range(n):
        if i == ref_id:
            continue
        i_ind = utils.id2ind(ref_id=ref_id, this_id=i)
        for j in range(n-1):
            j_id = utils.ind2id(ref_id=i, this_ind=j)
            j_ind = utils.id2ind(ref_id=ref_id, this_id=j_id)
            aux_vars = [th.SE3(base_pose_all[:, i, j], name=f"cross_edge_{i}_{j}")]
            w = th.DiagonalCostWeight(w_base[:, i, j, :3])
            if j_id == ref_id:
                ###### self rel pos cost ######
                optim_vars = [base_opted_pose[i_ind]]
                objective.add(th.AutoDiffCostFunction(optim_vars=optim_vars, err_fn=self_rel_pos_err_func, dim=3, aux_vars=aux_vars, name=f"self_rel_pos_cost_{i}_{j}", cost_weight=w))
            else:
                ###### cross rel pos cost ######
                optim_vars = base_opted_pose[i_ind], base_opted_pose[j_ind]
                objective.add(th.AutoDiffCostFunction(optim_vars=optim_vars, err_fn=cross_rel_pos_err_func, dim=3, aux_vars=aux_vars, name=f"cross_rel_pos_cost_{i}_{j}", cost_weight=w))

    ############ prior base pose cost ############
    for i in range(n-1):
        w = th.DiagonalCostWeight(w_base[:, ref_id, i, :])
        aux_targets = th.SE3(base_pose_all[:, ref_id, i], name=f"prior_node_{i}")
        objective.add(th.Difference(var=base_opted_pose[i], target=aux_targets, cost_weight=w, name=f"prior_pose_cost_{i}"))
    
    optimizer = th.LevenbergMarquardt(
        objective,
        max_iterations=5,
        step_size=1.0,
        linearization_cls=th.SparseLinearization,
        linear_solver_cls=th.CholmodSparseSolver,
        vectorize=True
    )

    theseus_optim = th.TheseusLayer(optimizer).to(device)
    theseus_inputs = generate_theseus_input(base_pose_all[:, ref_id, ...])
    # objective.update(theseus_inputs)
    solution, _ = theseus_optim.forward(theseus_inputs, optimizer_kwargs={"verbose": False})

    if get_pgo_info:
        feat_node, feat_base = get_opt_info(objective, n, m, node_opt_mask, node_aux_mask, base_mask_all, device=device)
        pgo_opt_info = {'node': feat_node, 'base': feat_base}
    else:
        pgo_opt_info = None

    out_node_pos = node_opt_pos.clone() # [bs, n-1, m, 3]
    out_base_pose = base_pose_all[:, ref_id, ...].clone() # [bs, n-1, 7]
    for i in range(n-1):
        pose_i = th.SE3(tensor=solution[f"opt_base_pose_{i}"])
        out_base_pose[:, i, :] = pose_i.to_x_y_z_quaternion()
        for j in range(m):
            exp = th.Point3(node_exparam[:, i, j, :])
            node_ij = pose_i.rotation().rotate(exp) + pose_i.translation()
            out_node_pos[:, i, j, :] = node_ij.tensor

    return out_node_pos, out_base_pose, pgo_opt_info


def run_pgo(tag_pos, tag_cov, tag_mask, tag_exparam, aux_pos, aux_mask, dis, dis_cov, base_pose, base_cov, base_mask, fix_weight=False, get_pgo_info=False, device='cuda:0'):
    '''
    input:
        tag_pos: torch.tensor, [bs, nf, n, n-1, m, 3]
        tag_cov: torch.tensor, [bs, nf, n, n-1, m, 3]
        tag_mask: torch.tensor, [bs, nf, n, n-1, m]
        tag_exparam: torch.tensor, [bs, nf, n, n-1, m, 3]
        aux_pos: torch.tensor, [bs, nf, n, m, 3]
        aux_mask: torch.tensor, [bs, nf, n, m]
        dis: torch.tensor, [bs, nf, n, n-1, m, m, 1]
        dis_cov: torch.tensor, [bs, nf, n, n-1, m, m, 1]
        base_pose: torch.tensor, [bs, nf, n, n-1, 7]
        base_cov: torch.tensor, [bs, nf, n, n-1, 6]
        base_mask: torch.tensor, [bs, nf, n, n-1]
        fix_weight: bool
        get_pgo_info: bool
        device: str, device id
    output:
        opted_tag_pos: [bs, nf, n, n-1, m, 3]
        opted_base_pose: [bs, nf, n, n-1, 7]
    bs: batch_size || nf: num of frames || n: num of robots || m: num of tags in each robot || mask means invalid nodes
    '''
    bs, nf, n, m = tag_pos.size(0), tag_pos.size(1), tag_pos.size(2), tag_pos.size(4)
    base_pose_xyzwxyz = utils.xyz_xyzw_2_xyz_wxyz(base_pose) # [bs, nf, n, n-1, 7]
    opted_node_pos = torch.zeros_like(tag_pos).to(device) # [bs, nf, n, n-1, m, 3]
    opted_base_pose = torch.zeros_like(base_pose_xyzwxyz).to(device) # [bs, nf, n, n-1, 7]
    for fi in range(nf):
        for ref_id in range(n):
            opted_node, opted_base, pgo_info = build_pgo(ref_id, tag_pos[:, fi, ref_id, ...], tag_cov[:, fi, ref_id, ...], tag_mask[:, fi, ref_id, ...],
                                                        tag_exparam[:, fi, ref_id, ...], aux_pos[:, fi, ref_id, ...], aux_mask[:, fi, ref_id, ...], 
                                                        dis[:, fi, ...], dis_cov[:, fi, ...], base_pose_xyzwxyz[:, fi, ...], 
                                                        base_cov[:, fi, ...], base_mask[:, fi, ...], fix_weight, get_pgo_info, device)
            opted_node_pos[:, fi, ref_id, ...] = opted_node
            opted_base_pose[:, fi, ref_id, ...] = opted_base
    opted_base_pose = utils.xyz_wxyz_2_xyz_xyzw(opted_base_pose)
    
    return opted_node_pos, opted_base_pose, pgo_info
