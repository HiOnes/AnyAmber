import argparse
import json
import os

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', 'y', '1'):
        return True
    if value in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def _load_config_file(path):
    if path is None:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        if path.endswith(".json"):
            return json.load(f)
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("YAML config requires PyYAML; use JSON or install PyYAML.") from exc
            return yaml.safe_load(f) or {}
    raise ValueError("Config file must be .json, .yaml, or .yml")

def get_args():
    parser = argparse.ArgumentParser()

    ######## Unified Parameters ########
    parser.add_argument('--config', type=str, default=None, help="Optional JSON/YAML config file; CLI values override it")
    parser.add_argument('--task', type=str, default="egat", choices=["match", "range", "egat", "end2end"])
    parser.add_argument('--mode', type=str, default="train", choices=["train", "eval", "infer"])
    parser.add_argument('--match_mode', type=str, default="compact", choices=["single", "compact", "split", "seq", "handcraft"], help="MatchNet variant to use")
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--shuffle', type=str2bool, default=False)
    parser.add_argument('--training', type=str2bool, default=False, help="True for training, False for inference or evaluation")
    parser.add_argument('--with_bearing', type=str2bool, default=False)
    parser.add_argument('--use_CPGO', type=str2bool, default=False)
    parser.add_argument('--use_RPGO', type=str2bool, default=False)
    parser.add_argument('--fix_PGO_weight', type=str2bool, default=False)
    parser.add_argument('--get_PGO_info', type=str2bool, default=False)
    parser.add_argument('--use_RF', type=str2bool, default=False, help="whether to use GRU range filter, if False, the RF output will be set to None")
    parser.add_argument('--train_RF', type=str2bool, default=False, help="whether to train and save GRU range filter weights when use_RF is True")
    parser.add_argument('--use_RF_sensor_embedding', type=str2bool, default=False, help="whether to use sensor embedding in GRU range filter")
    parser.add_argument('--robot_num', type=int, default=0, help="0: varying robots; 1: single robot; >1: multiple fixed robots")
    parser.add_argument('--anchor_num', type=int, default=9, help="number of anchors in single-robot scene, including padding nodes")
    parser.add_argument('--tag_num', type=int, default=1, help="number of tags in universal scene, single-robot[padding nodes included], multi-robot[padding nodes excluded]")
    parser.add_argument('--max_tag_num', type=int, default=4, help="number of max tags in universal scene, including padding nodes") 
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--gat_heads', type=int, default=3)
    parser.add_argument('--gat_layers', type=int, default=4)
    ######## Multi-Robot Parameters ########
    parser.add_argument('--max_cam_num', type=int, default=8)
    parser.add_argument('--others_embed_size', type=int, default=64)
    parser.add_argument('--timestamp_thres', type=float, default=2.5, help="timestamp gap larger than this value will be considered not continuous")
    parser.add_argument('--frame_win', type=int, default=40, help="multi-frame window size")
    parser.add_argument('--fixed_win', type=int, default=0, help="multi-frame fixed window size, should be smaller than frame_win")
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--max_eval_batches', type=int, default=0, help="0 means evaluate all batches")
    ######## Debugging Parameters ########
    parser.add_argument('--wrt_traj', type=str2bool, default=False, help="whether to write trajectory while inference under --model_file")
    parser.add_argument('--record_graph', type=str2bool, default=False, help="record generated EGAT graphs to CSV under --wrt_folder when supported")
    parser.add_argument('--record_uwb_seq', type=str2bool, default=False)
    parser.add_argument('--wrt_folder', type=str, default="./data/gnn/uni/my_recorded_graphs")
    parser.add_argument('--wrt_start_g_id', type=int, default=-1, help="start graph id when writing graphs to CSV")
    parser.add_argument('--wrt_ref_id', type=int, default=0, help="which robot to be the reference when writing traj, 0~robot_num-1")


    ######## Model and Dataset ########
    parser.add_argument('--load_pretrained_model', type=str2bool, default=False)
    parser.add_argument('--model_file', type=str, default="./checkpoints/uni/ref")
    parser.add_argument('--train_dataset', type=str, default="./data/gnn/uni/train_pretrain")
    parser.add_argument('--val_dataset', type=str, default="./data/gnn/uni/val_pretrain")
    parser.add_argument('--match_net_checkpoints', type=str, default="match-pos.pt")
    parser.add_argument('--egat_net_checkpoints', type=str, default="uni.pt")
    parser.add_argument('--rf_net_checkpoints', type=str, default="rf-embed.pt")

    # Params setting automatically by the system
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    
    config_args, _ = parser.parse_known_args()
    config = _load_config_file(config_args.config)
    if config:
        unknown_keys = sorted(set(config.keys()) - {action.dest for action in parser._actions})
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")
        parser.set_defaults(**config)

    args = parser.parse_args()
    return args
