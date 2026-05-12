from torch.utils.data import Dataset
import dgl
import torch
from copy import deepcopy

class ComPactedCSVDataset(dgl.data.CSVDataset):
    def __init__(self, robot_num, *args, **kwargs):
        super(ComPactedCSVDataset, self).__init__(*args, **kwargs)
        batchgraph_list = []
        batch_timestamp = torch.zeros(int(len(self.graphs)/robot_num), 2*robot_num, dtype=torch.int64)
        batch_ref_id = torch.zeros(int(len(self.graphs)/robot_num), 1*robot_num, dtype=torch.int8)
        batch_world_pose_delta = torch.zeros(int(len(self.graphs)/robot_num), 7*robot_num, dtype=torch.float32)
        batch_local2map = torch.zeros(int(len(self.graphs)/robot_num), 7*robot_num, dtype=torch.float32)
        for i in range(0, len(self.graphs), robot_num):
            batchgraph_list.append(dgl.batch(self.graphs[i:i + robot_num]))
            timestamp = torch.zeros(1, robot_num*2, dtype=torch.int64)
            ref_id = torch.zeros(1, robot_num, dtype=torch.int8)
            world_pose_delta = torch.zeros(1, robot_num*7, dtype=torch.float32)
            local2map = torch.zeros(1, robot_num*7, dtype=torch.float32)
            for j in range(robot_num):
                timestamp[0, 2*j:2*j+2] = self.data['timestamp'][i+j]
                if 'ref_id' in self.data:
                    ref_id[0, j] = self.data['ref_id'][i+j]
                else:
                    ref_id[0, j] = -1
                if 'world_pose_delta' in self.data:
                    world_pose_delta[0, 7*j:7*j+7] = self.data['world_pose_delta'][i+j]
                else:
                    world_pose_delta[0, 7*j:7*j+7] = torch.tensor([0,0,0,0,0,0,1], dtype=torch.float32)
                if 'local2map' in self.data:
                    local2map[0, 7*j:7*j+7] = self.data['local2map'][i+j]
                else:
                    local2map[0, 7*j:7*j+7] = torch.tensor([0,0,0,0,0,0,1], dtype=torch.float32)
            batch_timestamp[int(i/robot_num)] = timestamp
            batch_ref_id[int(i/robot_num)] = ref_id
            batch_world_pose_delta[int(i/robot_num)] = world_pose_delta
            batch_local2map[int(i/robot_num)] = local2map

        self.graphs = batchgraph_list
        self.data['timestamp'] = batch_timestamp
        self.data['ref_id'] = batch_ref_id
        self.data['world_pose_delta'] = batch_world_pose_delta
        self.data['local2map'] = batch_local2map
    
    def process(self):
        super(ComPactedCSVDataset, self).process()

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
 
    def __len__(self):
        return len(self.sequences)
 
    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return sequence, label
    
def create_continues_sequences(dataset, tw, timestamp_thres):
    inout_seq = []
    L = len(dataset)
    for i in range(L - tw):
        # validate if the data inside a seq is consecutive!!!
        seq = dataset[i:i + tw] # tuple(graph_list, msg_dict)
        graph_list, msg_dict = seq
        last_t = None
        is_consecutive = True
        for step in range(tw):
            # check timestamp
            if isinstance(msg_dict, dict):
                ts = msg_dict['timestamp']
            elif isinstance(msg_dict, torch.Tensor):
                ts = msg_dict
            else:
                raise ValueError("msg_dict must be a dict or a torch.Tensor containing 'timestamp' key.")
            t_decimal = float('0.' + str(ts[step,1].item())[1:])
            t = ts[step,0].item() + t_decimal
            if last_t is None:
                last_t = t
                continue
            if abs(t-last_t) > timestamp_thres:
                is_consecutive = False
                break
            last_t = t

        if is_consecutive:
            inout_seq.append(seq)
        
    return inout_seq

def create_continues_sequences_cam_match_reversed(dataset, args):
    inout_seq = []
    L = len(dataset)
    for i in range(L - args.frame_win):
        # validate if the data inside a seq is consecutive!!!
        seq = dataset[i:i + args.frame_win] # tuple(graph_list, msg_dict)
        graph_list_tmp, msg_dict = seq
        graph_list = deepcopy(graph_list_tmp)
        t_, label_cam_, label_others_ = None, None, None
        is_consecutive = True
        for step in reversed(range(args.frame_win)):
            # check timestamp
            t_decimal = float('0.' + str(msg_dict['timestamp'][step,1].item())[1:])
            t = msg_dict['timestamp'][step,0].item() + t_decimal
            label_cam = graph_list[step].ndata['label_match']['cam'] # [robot_num*max_cam_num, 1] 
            label_others = graph_list[step].ndata['label_match']['others'] # [robot_num*(robot_num-1), 1]
            label_src2des, label_des2src = torch.zeros_like(label_cam), torch.zeros_like(label_cam)
            label_cam = label_cam.reshape(args.robot_num, args.max_cam_num) # [robot_num, max_cam_num]
            label_others = label_others.reshape(args.robot_num, args.robot_num-1) # [robot_num, robot_num-1]
            if t_ is None:
                t_ = t
                label_cam_ = label_cam
                label_others_ = label_others
                graph_list[step].nodes['cam'].data['label_src2des'] = label_src2des # unused, but keep the same scheme
                graph_list[step].nodes['cam'].data['label_des2src'] = label_des2src # unused, but keep the same scheme
                continue
            if abs(t-t_) > args.timestamp_thres:
                is_consecutive = False
                break

            label_src2des, label_des2src = assign_frames(label_cam_, label_cam, label_others_, label_others)
            graph_list[step].nodes['cam'].data['label_src2des'] = label_src2des.reshape(args.robot_num*args.max_cam_num, 1)
            graph_list[step].nodes['cam'].data['label_des2src'] = label_des2src.reshape(args.robot_num*args.max_cam_num, 1)
            t_ = t

        if is_consecutive:
            inout_seq.append((graph_list, msg_dict))

    return inout_seq

def assign_frames(cam0, cam1, others0, others1):
    # input -- cam0, cam1: [robot_num, max_cam_num] | others0, others1: [robot_num, robot_num-1]
    # output -- cam0_to_cam1, cam1_to_cam0: [robot_num, max_cam_num]
    cam0_to_cam1, cam1_to_cam0 = torch.zeros_like(cam0), torch.zeros_like(cam1)
    for i in range(cam0.shape[0]):
        for j in range(cam0.shape[1]):
            if cam0[i,j] == -1:
                cam0_to_cam1[i,j] = -1
            else:
                cam0_to_cam1[i,j] = others1[i, cam0[i,j]]
            if cam1[i,j] == -1:
                cam1_to_cam0[i,j] = -1
            else:
                cam1_to_cam0[i,j] = others0[i, cam1[i,j]]
    return cam0_to_cam1, cam1_to_cam0
