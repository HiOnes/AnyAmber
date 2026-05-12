import os

import torch

import util.losses as losses
from runners.common import MetricMeter, build_dataloader, get_device, load_graph_dataset
from runners.egat_runner import (
    build_models,
    build_pgo_adapter,
    classify_robot_state,
    egat_step,
)
from util.graph_process import get_retain_nodes
from util.utils import get_T_from_quatpose, ind2id, trans_pos, trans_quatpose


def _timestamp_from_msg(msg):
    timestamp = msg["timestamp"]
    if timestamp.dim() == 3:
        stamp = timestamp[0, -1]
    else:
        stamp = timestamp[0]
    t_decimal = float("0." + str(stamp[1].item())[1:])
    return stamp[0].item() + t_decimal


def _local2map_from_msg(msg, args):
    local2map = msg["local2map"]
    if local2map.dim() == 3:
        local2map = local2map[:, -1, :]
    local2map = local2map.cpu().numpy().reshape(-1, 7)
    index = min(args.wrt_ref_id, local2map.shape[0] - 1)
    return get_T_from_quatpose(local2map[index], w_first=False)


def update_egat_prior(graph, msg, last_t, last_pred, args, device):
    graph = graph.to(device)
    nodes_retain = get_retain_nodes(graph, mode="flag")
    moving_keep, base_keep = nodes_retain["moving"], nodes_retain["base"]
    timestamp = _timestamp_from_msg(msg)
    is_consecutive = last_t is not None and abs(timestamp - last_t) <= args.timestamp_thres
    if last_pred is not None and is_consecutive:
        graph.nodes["base"].data["feat"][base_keep] = last_pred["base_pose"][-1]
        graph.nodes["moving"].data["feat"][moving_keep] = last_pred["moving_pos"][-1]
    else:
        graph.nodes["base"].data["feat"] = graph.ndata["label"]["base"]
        graph.nodes["moving"].data["feat"] = graph.ndata["label"]["moving"]
        print("reset prior")
    return timestamp, graph


def write_egat_traj(pred, target, transform, timestamp, path, args):
    os.makedirs(path, exist_ok=True)
    if args.robot_num > 1:
        pred_base = pred["base_pose"][-1].cpu().numpy().reshape(args.robot_num, args.robot_num - 1, 7)
        gt_base = target["base_pose"].cpu().numpy().reshape(args.robot_num, args.robot_num - 1, 7)
        for i in range(args.robot_num - 1):
            this_id = ind2id(args.wrt_ref_id, i)
            gt_map = trans_quatpose(gt_base[args.wrt_ref_id, i], transform, w_first=False)
            pred_map = trans_quatpose(pred_base[args.wrt_ref_id, i], transform, w_first=False)
            with open(os.path.join(path, f"gt{this_id}.txt"), "a") as f_gt:
                f_gt.write(" ".join(str(p) for p in [timestamp] + gt_map) + "\n")
            with open(os.path.join(path, f"pred{this_id}.txt"), "a") as f_pred:
                f_pred.write(" ".join(str(p) for p in [timestamp] + pred_map) + "\n")
        return

    gt_local = target["base_pose"].cpu().numpy().flatten()
    gt_map = trans_quatpose(gt_local, transform, w_first=False)
    is_single_tag = target["base_pose"].shape[0] == 1 and target["moving_pos"].shape[0] == 1
    if is_single_tag:
        pred_local = pred["moving_pos"][-1].cpu().numpy().flatten()
        pred_map = trans_pos(pred_local, transform) + [0, 0, 0, 1]
        gt_map = gt_map[:3] + [0, 0, 0, 1]
    else:
        pred_local = pred["base_pose"][-1].cpu().numpy().flatten()
        pred_map = trans_quatpose(pred_local, transform, w_first=False)
    with open(os.path.join(path, "gt.txt"), "a") as f_gt:
        f_gt.write(" ".join(str(p) for p in [timestamp] + gt_map) + "\n")
    with open(os.path.join(path, "UniAmber.txt"), "a") as f_pred:
        f_pred.write(" ".join(str(p) for p in [timestamp] + pred_map) + "\n")


def infer_egat(args):
    device = get_device(args)
    dataset = load_graph_dataset(args, args.val_dataset, compact=args.robot_num > 1)
    dataloader = build_dataloader(args, dataset, shuffle=False)
    robot_state = classify_robot_state(args)
    model_egat, model_rf, train_rf = build_models(args, device)
    model_egat.load_state_dict(torch.load(os.path.join(args.model_file, args.egat_net_checkpoints), map_location=device))
    if args.use_RF:
        model_rf.load_state_dict(torch.load(os.path.join(args.model_file, args.rf_net_checkpoints), map_location=device))
    model_egat.eval()
    model_rf.eval()
    set_loss = losses.SetLoss(args)
    pgo_func, get_pgo_input_func = build_pgo_adapter(robot_state, args, device)
    meter = MetricMeter()
    last_t, last_pred = None, None
    with torch.no_grad():
        for batch_idx, (graph, msg) in enumerate(dataloader, start=1):
            if args.max_eval_batches > 0 and batch_idx > args.max_eval_batches:
                break
            last_t, graph = update_egat_prior(graph, msg, last_t, last_pred, args, device)
            loss, last_pred, target = egat_step(
                robot_state,
                model_egat,
                model_rf,
                (graph, msg),
                set_loss,
                pgo_func,
                get_pgo_input_func,
                args,
                device,
            )
            meter.update(loss)
            if args.wrt_traj:
                write_egat_traj(last_pred, target, _local2map_from_msg(msg, args), last_t, args.model_file, args)
    metrics = meter.compute(rmse_keys=("pos", "tagpos"))
    print(
        "***EGAT Inference-- Loss {:.4f} | TagPos {:.4f} | Pos {:.4f} | Rot {:.4f} | Cov {:.4f}***".format(
            metrics.get("total", 0.0),
            metrics.get("tagpos", 0.0),
            metrics.get("pos", 0.0),
            metrics.get("rot", 0.0),
            metrics.get("cov", 0.0),
        )
    )
    return metrics
