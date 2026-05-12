import numpy as np
import torch
import transforms3d as tfs
import pytorch3d.transforms as p3dt
import networkx as nx
from typing import List
import math
from scipy.linalg import svd
from sklearn.decomposition import PCA

def id2ind(ref_id, this_id):
    if this_id < ref_id:
        return this_id
    return this_id - 1

def ind2id(ref_id, this_ind):
    if this_ind < ref_id:
        return this_ind
    return this_ind + 1

def set_pose_accuracy(pose: List[float]):
    ''' pose: List [x, y, z, qx, qy, qz, qw] / [x, y, z, qw, qx, qy, qz] '''
    p = pose.copy()
    for i in range(len(pose)):
        if i < 3:
            p[i] = round(pose[i], 3)
        else:
            p[i] = round(pose[i], 6)
    return p

def keep_w_positive(quat, w_first=False):
    assert quat.shape[-1] == 4, "Input quat should have shape [..., 4]"
    res = quat.clone()
    w_index = 0 if w_first else 3
    neg_row_ids = torch.where(quat[..., w_index]<0)
    res[neg_row_ids] = -quat[neg_row_ids]
    return res

def xyz_wxyz_2_xyz_xyzw(xyz_wxyz):
    '''
    xyz_wxyz: torch.tensor:[..., 7] / np.array:(7,) / list [*7]
    [x, y, z, qw, qx, qy, qz] -> [x, y, z, qx, qy, qz, qw]
    '''
    if isinstance(xyz_wxyz, np.ndarray) and xyz_wxyz.shape == (7,):
        new_indices = [0, 1, 2, 4, 5, 6, 3]
        return xyz_wxyz.copy()[new_indices]
    elif isinstance(xyz_wxyz, list) and len(xyz_wxyz) == 7:
        return xyz_wxyz[:3] + xyz_wxyz[4:] + [xyz_wxyz[3]]
    elif isinstance(xyz_wxyz, torch.Tensor) and xyz_wxyz.dim() > 1 and xyz_wxyz.shape[-1] == 7:
        return torch.cat((xyz_wxyz[..., :3], xyz_wxyz[..., 4:], xyz_wxyz[..., 3].unsqueeze(-1)), dim=-1)
    else:
        raise ValueError("Invalid Input in Function xyz_wxyz_2_xyz_xyzw")

def xyz_xyzw_2_xyz_wxyz(xyz_xyzw):
    '''
    xyz_xyzw: torch.tensor:[..., 7] / np.array:(7,) / list [*7]
    [x, y, z, qx, qy, qz, qw] -> [x, y, z, qw, qx, qy, qz]
    '''
    if isinstance(xyz_xyzw, np.ndarray) and xyz_xyzw.shape == (7,):
        new_indices = [0, 1, 2, 6, 3, 4, 5]
        return xyz_xyzw.copy()[new_indices]
    elif isinstance(xyz_xyzw, list) and len(xyz_xyzw) == 7:
        return xyz_xyzw[:3] + [xyz_xyzw[6]] + xyz_xyzw[3:6]
    elif isinstance(xyz_xyzw, torch.Tensor) and xyz_xyzw.dim() > 1 and xyz_xyzw.shape[-1] == 7:
        return torch.cat((xyz_xyzw[..., :3], xyz_xyzw[..., 6].unsqueeze(-1), xyz_xyzw[..., 3:6]), dim=-1)
    else:
        raise ValueError("Invalid Input in Function xyz_wxyz_2_xyz_xyzw")

def eulerPose_adding(pose0, pose_delta):
    '''
    pose0: [x, y, z, roll, pitch, yaw]
    pose_delta: [dx, dy, dz, droll, dpitch, dyaw]
    '''
    R0 = tfs.euler.euler2mat(pose0[3], pose0[4], pose0[5], 'sxyz')
    R_delta = tfs.euler.euler2mat(pose_delta[3], pose_delta[4], pose_delta[5], 'sxyz')
    R1 = np.dot(R0, R_delta)
    euler1 = torch.tensor(tfs.euler.mat2euler(R1, 'sxyz'))
    pos1 = np.array(pose0[:3], dtype=float).reshape(3,1) + np.array(pose_delta[:3], dtype=float).reshape(3,1)
    return np.concatenate((pos1, euler1), axis=0).flatten()

def eulerPose_adding_torch(pose0, pose_delta):
    '''
    pose0: [x, y, z, roll, pitch, yaw]
    pose_delta: [dx, dy, dz, droll, dpitch, dyaw]
    '''
    R0 = p3dt.euler_angles_to_matrix(pose0[3:6], 'XYZ')
    R_delta = p3dt.euler_angles_to_matrix(pose_delta[3:6], 'XYZ')
    R1 = torch.mm(R0, R_delta)
    euler1 = p3dt.matrix_to_euler_angles(R1, 'XYZ')
    pos1 = pose0[:3] + pose_delta[:3]
    return torch.concat((pos1, euler1), dim=0).reshape(1,6)
    
def batched_pos_R_to_T_torch(batch_pos, batch_R):
    '''
    batch_pos: [batch, 3]
    batch_R: [batch, 3, 3]
    '''
    last_line = torch.tensor([[[0, 0, 0, 1]]]).repeat(batch_pos.shape[0], 1, 1).to(batch_pos.device)
    return torch.cat((torch.cat((batch_R, batch_pos.reshape(-1,3,1)), dim=2), last_line), dim=1)

def batched_eulerPose_adding_torch(batch_pose0, batch_pose_delta, return_euler=True):
    '''
    batch_pose0: [batch, 6]
    batch_pose_delta: [batch, 6]
    '''
    batch_R0 = p3dt.euler_angles_to_matrix(batch_pose0[:,3:6], 'XYZ')
    batch_R_delta = p3dt.euler_angles_to_matrix(batch_pose_delta[:,3:6], 'XYZ')
    batch_R1 = torch.bmm(batch_R0, batch_R_delta)
    batch_pos1 = batch_pose0[:,0:3] + batch_pose_delta[:,0:3]
    if return_euler:
        batch_euler1 = p3dt.matrix_to_euler_angles(batch_R1, 'XYZ')
        return torch.cat((batch_pos1, batch_euler1), dim=1)
    return batch_pos1, batch_R1
    
def batched_T_to_eulerPose_torch(batch_T):
    '''
    batch_T: [batch, 4, 4]
    '''
    batch_R = batch_T[:, :3, :3]
    batch_euler = p3dt.matrix_to_euler_angles(batch_R, 'XYZ')
    batch_pos = batch_T[:, :3, 3]
    return torch.cat((batch_pos, batch_euler), dim=1)

# def batched_T_to_quatPose_torch(batch_T, w_first):
#     """
#     Input:
#         batch_T: torch.tensor, [batch, 4, 4]
#     Output:
#         batch_pose: torch.tensor, [batch, 7]
#     """
#     batch_pos = batch_T[:, :3, 3]
#     batch_quat = p3dt.matrix_to_quaternion(batch_T[:, :3, :3])
#     batch_quatPose = torch.cat((batch_pos, batch_quat), dim=1)
#     if not w_first:
#         batch_quatPose = xyz_wxyz_2_xyz_xyzw(batch_quatPose)
#     return batch_quatPose

# def batched_quatPose_to_T_torch(batch_quatPose, w_first):
#     '''
#     Input: 
#         batch_quatpose: torch.tensor, [batch, 7]
#         w_first: bool
#     Output:
#         T: torch.tensor, [batch, 4, 4]
#     '''
#     if not w_first:
#         batch_quatPose = xyz_xyzw_2_xyz_wxyz(batch_quatPose)
#     t = batch_quatPose[:, 0:3]
#     R = p3dt.quaternion_to_matrix(batch_quatPose[:,3:7])
#     T = batched_pos_R_to_T_torch(t, R)
#     return T

def batched_quat_to_euler_torch(batch_quat, w_first=True):
    '''
    batch_quat: [batch, 4]
    '''
    if not w_first:
        batch_quat = torch.cat((batch_quat[:, 3].unsqueeze(1), batch_quat[:, :3]), dim=1)
    batch_R = p3dt.quaternion_to_matrix(batch_quat)
    batch_euler = p3dt.matrix_to_euler_angles(batch_R, 'XYZ')
    return batch_euler

def batched_quat_diff_torch(batch_quat0, batch_quat1, w_first=True):
    '''
    # batch_quat0: [batch, 4]
    # batch_quat1: [batch, 4]
    '''
    if not w_first:
        batch_quat0 = torch.cat((batch_quat0[:, 3].unsqueeze(1), batch_quat0[:, :3]), dim=1)
        batch_quat1 = torch.cat((batch_quat1[:, 3].unsqueeze(1), batch_quat1[:, :3]), dim=1)
    batch_quat0 = keep_w_positive(batch_quat0)
    batch_quat1 = keep_w_positive(batch_quat1)
    batch_quat_diff = p3dt.quaternion_multiply(batch_quat0, p3dt.quaternion_invert(batch_quat1))
    return batch_quat_diff

def batched_quatPose_update_torch(pose_ref_delta, pose_ref_k_, pose_k_delta, w_first):
    '''
    inputs:
        pose_ref_delta: torch.Tensor, [batch, 7]
        pose_ref_k_: torch.Tensor, [batch, 7]
        pose_k_delta: torch.Tensor, [batch, 7]
        w_first: bool
    outputs:
        pose_ref_k: torch.Tensor, [batch, 7] 
    '''
    T_ref_delta, T_ref_k_, T_k_delta = [get_T_from_quatpose(p, w_first) for p in (pose_ref_delta, pose_ref_k_, pose_k_delta)]
    T_ref_k = torch.bmm(torch.inverse(T_ref_delta), torch.bmm(T_ref_k_, T_k_delta))  
    pose_ref_k = get_quatpose_from_T(T_ref_k, w_first)
    return pose_ref_k

def batched_deltaT_from_quatpose(pose0, pose1, w_first):
    '''
    input:
        pose0 & pose1: [bs, 7] (x, y, z, qw, qx, qy, qz) / (x, y, z, qx, qy, qz, qw)
        w_first: bool
    output:
        T_delta: [bs, 4, 4]
    '''
    T0, T1 = [get_T_from_quatpose(p, w_first) for p in (pose0, pose1)]
    T_delta = torch.bmm(torch.inverse(T0), T1)
    return T_delta

def batched_eulerPose_to_T_torch(batch_eulerPose):
    '''
    batch_eulerPose: [batch, 6]
    '''
    batch_R = p3dt.euler_angles_to_matrix(batch_eulerPose[:,3:6], 'XYZ')
    batch_pos = batch_eulerPose[:,0:3]
    last_line = torch.tensor([[[0, 0, 0, 1]]]).repeat(batch_eulerPose.shape[0], 1, 1).to(batch_eulerPose.device)
    return torch.cat((torch.cat((batch_R, batch_pos.reshape(-1,3,1)), dim=2), last_line), dim=1)

def batched_imu_base_to_map_torch(batch_imu_base, batch_T_base_to_map):
    '''
    batch_imu_base: [batch, seq_len, 6] [wx, wy, wz, ax, ay, az]
    batch_T_base_to_map: [batch, 4, 4]
    '''
    batch_g_map = torch.tensor([0, 0, 9.8]).reshape(1,1,3,1).repeat(batch_imu_base.shape[0], batch_imu_base.shape[1], 1, 1).to(batch_imu_base.device)
    batch_R_base_to_map = batch_T_base_to_map[:, :3, :3].unsqueeze(1).repeat(1,batch_imu_base.shape[1],1,1)
    batch_vw_base = batch_imu_base[:,:,0:3].unsqueeze(-1)
    batch_acc_base = batch_imu_base[:,:,3:6].unsqueeze(-1)
    batch_vw_map = torch.matmul(batch_R_base_to_map, batch_vw_base)
    batch_acc_map = torch.matmul(batch_R_base_to_map, batch_acc_base) - batch_g_map
    res = torch.cat((batch_vw_map.squeeze(-1), batch_acc_map.squeeze(-1)), dim=2)
    return res

def batched_eulerPose_3dim_to_6dim_torch(batch_eulerPose_3dim):
    '''
    batch_eulerPose_3dim: [batch, 3] x, y, yaw
    '''
    z_r_p = torch.zeros(batch_eulerPose_3dim.shape[0], 3).to(batch_eulerPose_3dim.device)
    pose_6dim = torch.cat((batch_eulerPose_3dim[:,:2], z_r_p, batch_eulerPose_3dim[:,2].reshape(-1,1)), dim=1)
    return pose_6dim

def batched_eulerPose_6dim_to_3dim_torch(batch_eulerPose_6dim):
    '''
    batch_eulerPose_6dim: [batch, 6] x, y, z, roll, pitch, yaw
    '''
    pose_3dim = torch.cat((batch_eulerPose_6dim[:,:2], batch_eulerPose_6dim[:,-1].unsqueeze(1)), dim=1)
    return pose_3dim

def get_quatpose_from_eulerpose(eulerpose, is_degree, w_first):
    '''
    input:
        eulerpose: list [x, y, z, roll, pitch, yaw]
        is_degree: bool
        w_first: bool
    output:
        quatpose: list [x, y, z, qx, qy, qz, qw]
    '''
    translation = eulerpose[:3]
    [roll, pitch, yaw] = eulerpose[3:]
    if is_degree:
        [roll, pitch, yaw] = [math.radians(angle) for angle in [roll, pitch, yaw]]
    quat = tfs.euler.euler2quat(roll, pitch, yaw, axes='sxyz')
    quatpose = translation + quat.tolist() # w_first
    if not w_first:
        quatpose = xyz_wxyz_2_xyz_xyzw(quatpose)
    return quatpose

def get_T_from_quatpose(pose, w_first):
    """
    input:
        pose: torch.tensor[..., 7]  or list[*7]
        w_first: bool
    output:
        T: torch.tensor[..., 4, 4] or np.array[4, 4]
    """
    if not w_first:
        pose = xyz_xyzw_2_xyz_wxyz(pose)
    if isinstance(pose, torch.Tensor):
        t = pose[:, 0:3]
        R = p3dt.quaternion_to_matrix(pose[:,3:7])
        T = batched_pos_R_to_T_torch(t, R)
    else:
        R = tfs.quaternions.quat2mat(pose[3:])
        t = np.array(pose[:3], dtype=float).reshape(3,1)
        T_tmp = np.concatenate((R, t), axis=1)
        last_line = np.array([0, 0, 0, 1]).reshape(1,4)
        T = np.concatenate((T_tmp, last_line), axis=0)
    return T

def get_quatpose_from_T(T, w_first):
    """
    input:
        T: torch.tensor[..., 4, 4] or np.array[4, 4]
        w_first: bool
    output:
        pose: torch.tensor[..., 7]  or list[*7]
    """
    if isinstance(T, torch.Tensor):
        t = T[:, :3, 3]
        q = p3dt.matrix_to_quaternion(T[:, :3, :3])
        pose = torch.cat((t, q), dim=1)
    else:
        q = tfs.quaternions.mat2quat(T[:3, :3]).tolist()
        t = T[:3, -1].tolist()
        pose = t+q
    if not w_first:
        pose = xyz_wxyz_2_xyz_xyzw(pose)
    return pose

def get_T_from_eulerpose(pose):
    '''
    input:
        pose: [x, y, z, roll, pitch, yaw]
    output:
        T: np.array [4, 4]
    '''
    R = tfs.euler.euler2mat(pose[3], pose[4], pose[5], 'sxyz')
    t = np.array(pose[:3], dtype=float).reshape(3,1)
    T_tmp = np.concatenate((R, t), axis=1)
    last_line = np.array([0, 0, 0, 1]).reshape(1,4)
    T = np.concatenate((T_tmp, last_line), axis=0)
    return T

def trans_eulerpose(pose, T_local0_to_local1):
    '''
    pose: [x, y, z, roll, pitch, yaw]
    '''
    T_base_to_local0 = get_T_from_eulerpose(pose)
    T_base_to_local1 = np.dot(T_local0_to_local1, T_base_to_local0)
    R_base_to_local1 = T_base_to_local1[:3,:3]
    pos_base_to_local1 = T_base_to_local1[:3,3]
    euler_base_to_local1 = tfs.euler.mat2euler(R_base_to_local1, 'sxyz')
    return np.concatenate((pos_base_to_local1, euler_base_to_local1), axis=0).flatten().tolist()

def trans_quatpose(pose, T, w_first, return_quatpose=True):
    """
    Input:
        pose: torch.tensor[..., 7]  or list[*7]
        T: torch.tensor[..., 4, 4] or np.array[4, 4]
        w_first: bool
        return_quatpose: bool, if False, return T instead of quatpose
    Output:
        out_pose: 
            return_quatpose: torch.tensor[..., 7]  or list[*7]
            return_T: torch.tensor[..., 4, 4] or np.array[4, 4]
    """
    if isinstance(pose, torch.Tensor):
        assert isinstance(T, torch.Tensor) and pose.shape[0] == T.shape[0]
        T0 = get_T_from_quatpose(pose, w_first)
        T1 = torch.bmm(T, T0)
    else:
        assert len(pose) == 7 and isinstance(T, np.ndarray)
        T0 = get_T_from_quatpose(pose, w_first)
        T1 = np.dot(T, T0)
    if not return_quatpose:
        return T1
    out_pose = get_quatpose_from_T(T1, w_first)
    return out_pose

def trans_pos(pos, T):
    """
    Input:
        pos: torch.tensor[..., 3]  or list[*3]
        T: torch.tensor[..., 4, 4] or np.array[4, 4]
    Output:
        out_pos: torch.tensor[..., 3]  or list[*3]
    """
    if isinstance(pos, torch.Tensor):
        assert isinstance(T, torch.Tensor) and pos.shape[0] == T.shape[0]
        out_pos = torch.bmm(T[..., :3, :3], pos.unsqueeze(-1)).squeeze(-1) + T[..., :3, 3]
    else:
        assert len(pos) == 3 and isinstance(T, np.ndarray)
        out_pos = np.dot(T[:3,:3], np.array(pos).reshape(3,1)) + T[:3,3].reshape(3,1)
        out_pos = out_pos.flatten().tolist()
    return out_pos

def inv_quatpose(pose, w_first):
    """
    Input:
        pose: torch.tensor[..., 7]  or list[*7]
        w_first: bool
    Output:
        out_pose: torch.tensor[..., 7]  or list[*7]
    """
    T = get_T_from_quatpose(pose, w_first)
    if isinstance(T, torch.Tensor):
        T_inv = torch.inverse(T)
    else:
        T_inv = np.linalg.inv(T)
    out_pose = get_quatpose_from_T(T_inv, w_first)
    return out_pose

def accumulate_quatpose(pose0, pose1, w_first):
    """
    Input:
        pose0 & pose1: torch.tensor[..., 7]  or list[*7]
        w_first: bool
    Output:
        out_pose: torch.tensor[..., 7]  or list[*7]
    """
    T0 = get_T_from_quatpose(pose0, w_first)
    out_pose = trans_quatpose(pose1, T0, w_first, return_quatpose=True)
    return out_pose


def get_deltaT_from_quatpose(pose0, pose1, w_first):
    '''
    input:
        pose0 & pose1: (x, y, z, qw, qx, qy, qz) / (x, y, z, qx, qy, qz, qw)
        w_first: bool
    output:
        T_delta: np.array [4, 4]
    '''
    T0, T1 = get_T_from_quatpose(pose0, w_first), get_T_from_quatpose(pose1, w_first)
    T_delta = np.dot(np.linalg.inv(T0), T1)
    return T_delta

def pca(X, out_dim):
    """
    input:
        X: np.array, [m, n], m: num of samples, n: feature dimension
        out_dim: int
    
    output:
        Uk: projection matrix, [n, out_dim]
    """
    m, n = X.shape
    
    # 1. Data centralized
    X_centered = X - np.mean(X, axis=0)
    
    # 2. Calculate covariance matrix
    Sigma = (1.0 / m) * np.dot(X_centered.T, X_centered)
    
    # 3. SVD
    U, S, Vt = svd(Sigma)
    
    # 4. Get principal columns
    Uk = U[:, :out_dim]
    
    return Uk

def get_nominal_transformation(anchor_array):
    """
    input:
        anchor_array: np.array, [n, 3]
    output:
        T: np.array, [4, 4]
    """
    t = np.mean(anchor_array, axis=0).reshape(3,1)
    R = pca(anchor_array, out_dim=3)
    T_tmp = np.concatenate((R, t), axis=1)
    last_line = np.array([0, 0, 0, 1]).reshape(1,4)
    T = np.concatenate((T_tmp, last_line), axis=0)
    return T

def show_graph(graph, nodeFeat, EdgeFeat):
    '''
    eg. show_graph(g, "desc", "weights")
    '''
    G = graph.to_networkx(node_attrs=nodeFeat.split(), edge_attrs=EdgeFeat.split()) # transform dgl graph to networks
    pos = nx.spring_layout(G)
    nx.draw(G, pos, edge_color="grey", node_size=500, with_labels=True, connectionstyle="arc3,rad=0.1") # plot graph
    node_data = nx.get_node_attributes(G, nodeFeat)
    node_labels = { key: "N:" + ','.join([str(round(x,3)) for x in node_data[key].numpy()]) for key in node_data }
    pos_higher = {}
    for k, v in pos.items():  # adjust the position of the vertex
        if(v[1]>0):
            pos_higher[k] = (v[0]-0.04, v[1]+0.04)
        else:
            pos_higher[k] = (v[0]-0.04, v[1]-0.04)
    nx.draw_networkx_labels(G, pos_higher, labels=node_labels, font_color="brown", font_size=12) # display the features on the vertex
    edge_labels = nx.get_edge_attributes(G, EdgeFeat)
    edge_labels= { (key[0],key[1]): "E" + str(key[0]) + "-" + str(key[1]) + ": " + str(round(edge_labels[key].item(), 3)) for key in edge_labels }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, alpha=0.5) # display the features on the edges

def write_to_file(file_path, data, last_dim_num):
    ''' 
    data: [bs, n, n-1, last_dim_nim]
    '''
    with open(file_path, 'w') as f:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    for l in range(last_dim_num):
                        f.write(str(data[i,j,k,l].item()))
                        f.write(' ')
                    f.write('\n')
                f.write('\n')
            f.write('----------------------------------------------------------------\n')

def compute_solution_consistency(solutions):
    """
    solutions: (K,N,3)
        K = num of diffusion samples
        N = num of predicted robots
        3 = (x,y,z)
    return dict metrics
    """

    K,N,_ = solutions.shape

    # # gauge fix
    # aligned = []
    # for k in range(K):
    #     aligned.append(gauge_fix_positions(solutions[k]))

    # aligned = torch.stack(aligned)  # (K,N,3)
    aligned = solutions

    # mean solution
    mean_sol = aligned.mean(dim=0)

    # deviation
    diff = aligned - mean_sol

    per_robot_var = torch.mean(torch.sum(diff**2, dim=-1), dim=0)

    total_var = torch.mean(per_robot_var)

    # # pairwise distance
    # pairwise = []

    # for i in range(K):
    #     for j in range(i+1, K):

    #         d = torch.norm(aligned[i] - aligned[j], dim=-1).mean()

    #         pairwise.append(d)

    # pairwise = torch.tensor(pairwise)

    return {
        "total_variance": total_var.item(),
        "per_robot_variance": per_robot_var,
        # "mean_pairwise_distance": pairwise.mean().item(),
        # "max_pairwise_distance": pairwise.max().item(),
    }

def compute_solution_rank(solutions):

    K,N,_ = solutions.shape

    x = solutions.reshape(K,-1)

    x = x - x.mean(0)

    cov = x.T @ x / K

    eigvals = torch.linalg.eigvals(cov).real

    rank = (eigvals > 1e-6).sum()

    return rank

def pca_projection(solutions, n_components=2):
    """
    Input:
        solutions: [K, 3]
    Output:
        y: [K, n_components]
    """

    pca = PCA(n_components=n_components)

    y = pca.fit_transform(solutions)

    return y

def generate_noise_schedule(num_frames, mean_noise, noise_range):
    """
    为每一帧生成递增的整数噪声系数
    
    Args:
        num_frames (int): 总帧数
        mean_noise (float): 期望噪声平均值
        noise_range (tuple): (min_noise, max_noise)
    
    Returns:
        List[int]: 每一帧的噪声系数
    """
    min_noise, max_noise = noise_range
    
    # Step 1: 生成一个线性递增的基础序列（0~1）
    base = np.linspace(0, 1, num_frames)
    
    # Step 2: 拉伸到指定范围
    noise = min_noise + base * (max_noise - min_noise)
    
    # Step 3: 加入随机扰动（保证趋势但不完全线性）
    noise += np.random.randn(num_frames) * (max_noise - min_noise) * 0.05
    
    # Step 4: 调整均值到目标 mean_noise
    current_mean = noise.mean()
    noise += (mean_noise - current_mean)
    
    # Step 5: clip 到合法范围
    noise = np.clip(noise, min_noise, max_noise)
    
    # Step 6: 强制单调递增（关键步骤）
    noise = np.maximum.accumulate(noise)
    
    # Step 7: 转成整数
    noise = np.round(noise).astype(int)
    
    return noise.tolist()