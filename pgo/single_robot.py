import torch
import theseus as th
import util.utils as utils
import torch.nn.functional as F

def generate_theseus_input(node_opt_pos, base_opt_pose):
    '''
    input:
        node_opt_pos: torch.tensor, [bs, n, 3] tag
        node_opt_mask: torch.tensor, [bs, n]
        base_opt_pose: torch.tensor, [bs, 7] uav
    '''
    node_opt_pos_clone = node_opt_pos.clone()
    base_opt_pose_clone = base_opt_pose.clone()
    inputs = {}
    for i in range(node_opt_pos_clone.size(1)):
        inputs[f"opt_node_pos_{i}"] = th.Point3(node_opt_pos_clone[:, i, :])
    inputs["opt_base_pose"] = th.SE3(base_opt_pose_clone)
    return inputs

def dis_err_func(optim_vars, aux_vars):
    [node_i] = optim_vars
    node_j, dis = aux_vars
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

def build_pgo(node_opt_pos, node_opt_cov, node_opt_mask, node_exparam, node_aux_pos, node_aux_mask, dis_edge, dis_edge_cov, base_opt_pose, base_opt_cov, base_opt_mask, fix_weight, get_pgo_info, device='cuda:0'):
    '''
    input:
        node_opt_pos: torch.tensor, [bs, n, 3] tag
        node_opt_cov: torch.tensor, [bs, n, 3]
        node_opt_mask: torch.tensor, [bs, n]
        node_exparam: torch.tensor, [bs, n, 3]
        node_aux_pos: torch.tensor, [bs, m, 3] anchor
        node_aux_mask: torch.tensor, [bs, m]
        dis_edge: torch.tensor, [bs, n, m]
        dis_edge_cov: torch.tensor, [bs, n, m, 1]
        base_opt_pose: torch.tensor, [bs, 7] uav
        base_opt_cov: torch.tensor, [bs, 6]
        base_opt_mask: torch.tensor, [bs, 1]
        device: str, device id
    output:
        out_node_pos: torch.tensor, [bs, n, 3]
        out_base_pose: torch.tensor, [bs, 7]
        pgo_info: Optional, dict containing node and base pose optimization info
    '''
    bs, n, m = node_opt_pos.size(0), node_opt_pos.size(1), node_aux_pos.size(1)
    # assert bs == 1 
    objective = th.Objective().to(device)
    node_opted_pos = { i : th.Point3(node_opt_pos[:, i, :], name=f"opt_node_pos_{i}") for i in range(n)}
    base_opted_pose = th.SE3(base_opt_pose, name="opt_base_pose")
    
    #### weights ####
    if fix_weight:
        w_opt = torch.ones((bs, n, 3), device=device)
        w_dis = torch.ones((bs, n, m, 1), device=device)
        w_base = torch.ones((bs, 6), device=device)
    else:
        w_opt, w_dis, w_base = [torch.sqrt(1.0/(cov*10.0) + 1e-6) for cov in (node_opt_cov, dis_edge_cov, base_opt_cov)]

    ################# dis pair cost ###############
    for i in range(n):
        for j in range(m):
            w = th.ScaleCostWeight(w_dis[:, i, j, :])
            optim_vars = [node_opted_pos[i]]
            aux_vars = [th.Point3(node_aux_pos[:, j, :]), th.Variable(dis_edge[:, i, j].unsqueeze(1))]
            objective.add(th.AutoDiffCostFunction(optim_vars=optim_vars, err_fn=dis_err_func, dim=1, aux_vars=aux_vars, name=f"dis_cost_{i}_{j}", cost_weight=w))
    
    ################# node exparam cost ###############
    for i in range(n):
        # w = th.DiagonalCostWeight(w_opt[:, i, :])
        w = torch.ones((bs, 3), device=device)
        w[node_opt_mask[:, i]] = 1e-6
        w[base_opt_mask[:, 0]] = 1e-6
        w = th.DiagonalCostWeight(w)
        optim_vars = [node_opted_pos[i], base_opted_pose]
        aux_vars = [th.Point3(node_exparam[:, i, :])]
        objective.add(th.AutoDiffCostFunction(optim_vars=optim_vars, err_fn=exparam_err_func, dim=3, aux_vars=aux_vars, name=f"exparam_cost_{i}", cost_weight=w))
    
    ############ prior node pos cost ############
    for i in range(n):
        w = th.DiagonalCostWeight(w_opt[:, i, :])
        # w = th.DiagonalCostWeight(torch.ones((bs, 3), device=device))
        aux_targets = th.Point3(node_opt_pos[:, i, :])
        objective.add(th.Difference(var=node_opted_pos[i], target=aux_targets, name=f"prior_pos_cost_{i}", cost_weight=w))

    ############ prior base pose cost ############
    aux_targets = th.SE3(base_opt_pose)
    w = th.DiagonalCostWeight(w_base)
    objective.add(th.Difference(var=base_opted_pose, target=aux_targets, name=f"prior_pose_cost", cost_weight=w))
    
    optimizer = th.LevenbergMarquardt(
        objective,
        max_iterations=5,
        step_size=1.0,
        linearization_cls=th.SparseLinearization,
        linear_solver_cls=th.CholmodSparseSolver,
        vectorize=True
    )

    theseus_optim = th.TheseusLayer(optimizer).to(device)
    theseus_inputs = generate_theseus_input(node_opt_pos, base_opt_pose)
    # objective.update(theseus_inputs)
    solution, _ = theseus_optim.forward(theseus_inputs, optimizer_kwargs={"verbose": False})

    if get_pgo_info:
        feat_node, feat_base = get_opt_info(objective, n, m, node_opt_mask, node_aux_mask, base_opt_mask, device=device)
        pgo_opt_info = {'node': feat_node, 'base': feat_base}
    else:
        pgo_opt_info = None

    out_node_pos = node_opt_pos.clone()
    for i in range(n):
        out_node_pos[:, i, :] = th.Point3(tensor=solution[f"opt_node_pos_{i}"]).tensor
    out_base_pose = th.SE3(tensor=solution[f"opt_base_pose"]).to_x_y_z_quaternion()

    return out_node_pos, out_base_pose, pgo_opt_info


def run_pgo(node_opt_pos, node_opt_cov, node_opt_mask, node_exparam, node_aux_pos, node_aux_mask, dis_edge, dis_edge_cov, base_opt_pose, base_opt_cov, base_opt_mask, fix_weight=False, get_pgo_info=False, device='cuda:0'):
    '''
    input:
        node_opt_pos: [batch_size, num_frame, n, 3]
        node_opt_cov: [batch_size, num_frame, n, 3]
        node_opt_mask: [batch_size, num_frame, n]
        node_exparam: [batch_size, num_frame, n, 3]
        node_aux_pos: [batch_size, num_frame, m, 3]
        node_aux_mask: [batch_size, num_frame, m]
        dis_edge: [batch_size, num_frame, n, m]
        dis_edge_cov: [batch_size, num_frame, n, m, 1]
        base_opt_pose: [batch_size, num_frame, 7]
        base_opt_cov: [batch_size, num_frame, 6]
        base_opt_mask: [batch_size, num_frame, 1]
    output:
        opted_node_pos: [batch_size, num_frame, n, 3]
        opted_base_pose: [batch_size, num_frame, 7]
        pgo_losses: [1] unused
    '''
    bs, n_f, n_opt, n_aux = node_opt_pos.size(0), node_opt_pos.size(1), node_opt_pos.size(2), node_aux_pos.size(2)
    assert node_opt_pos.size(-1) == 3
    base_pose_xyzwxyz = utils.xyz_xyzw_2_xyz_wxyz(base_opt_pose) # [bs, n_f, 7]
    opted_node_pos = torch.zeros_like(node_opt_pos).to(device) # [bs, n_f, n, 3]
    opted_base_pose = torch.zeros_like(base_pose_xyzwxyz).to(device) # [bs, n_f, 7]
    # pgo_losses = 0
    for fi in range(n_f):
        opted_node, opted_base, pgo_opt_info = build_pgo(node_opt_pos[:, fi, :, :], node_opt_cov[:, fi, :, :], node_opt_mask[:, fi, :],
                                                     node_exparam[:, fi, :, :], node_aux_pos[:, fi, :, :], node_aux_mask[:, fi, :], 
                                                     dis_edge[:, fi, :, :], dis_edge_cov[:, fi, :, :], base_pose_xyzwxyz[:, fi, :], 
                                                     base_opt_cov[:, fi, :], base_opt_mask[:, fi, :], fix_weight, get_pgo_info, device)
        opted_node_pos[:, fi, :, :] = opted_node
        opted_base_pose[:, fi, :] = opted_base
        # pgo_losses += pgo_loss.mean()
    opted_base_pose = utils.xyz_wxyz_2_xyz_xyzw(opted_base_pose)
    
    return opted_node_pos, opted_base_pose, pgo_opt_info
