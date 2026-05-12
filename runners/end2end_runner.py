import os
from queue import Queue

import torch

import util.losses as losses
from pgo import multi_robot as pgo_multi
from model.range_filter import RangeFilter
from model.match_single_egat import MatchSingleEgat
from model.match_compact_egat import MatchCompactEgat
from model.egat import Egat
from runners.common import MetricMeter, build_dataloader, get_device, load_graph_dataset, load_sequence_dataset
from runners.egat_runner import apply_range_filter, get_input_format
from util.graph_process import get_pgo_input_multi, get_retain_nodes
from util.utils import get_T_from_quatpose, ind2id, trans_quatpose


END2END_KEYS = ("total", "match", "precision", "recall", "pos", "rot", "cov")


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


def _print_end2end_metrics(metrics, title):
    body = " | ".join(f"{key} {metrics.get(key, 0.0):.4f}" for key in END2END_KEYS)
    print(f"***{title}-- {body}***")


def _ensure_end2end_args(args):
    if args.robot_num <= 1:
        raise ValueError("end2end expects a fixed multi-robot compact dataset.")
    if args.match_mode not in ("single", "compact"):
        raise ValueError("end2end supports --match_mode single or compact.")
    if args.record_graph and args.match_mode != "single":
        raise ValueError("end2end --record_graph is currently supported only with --match_mode single.")


def _build_end2end_backend(args, device, match_model_cls):
    model_match = match_model_cls(args, device).to(device)
    n_in_dict, e_in_dict = get_input_format(args)
    model_egat = Egat(n_in_dict, e_in_dict, args.embed_dim, args.gat_heads, args.gat_layers).to(device)
    model_rf = RangeFilter(max_num=64, hidden_size=args.embed_dim, device=device).to(device)
    model_match.load_state_dict(torch.load(os.path.join(args.model_file, args.match_net_checkpoints), map_location=device))
    model_egat.load_state_dict(torch.load(os.path.join(args.model_file, args.egat_net_checkpoints), map_location=device))
    if args.use_RF:
        model_rf.load_state_dict(torch.load(os.path.join(args.model_file, args.rf_net_checkpoints), map_location=device))
    model_match.eval()
    model_egat.eval()
    model_rf.eval()
    return model_match, model_egat, model_rf


def _single_match_cam_target(graph):
    if "cam" not in graph.ntypes or "label_match" not in graph.nodes["cam"].data:
        return None
    return graph.ndata["label_match"]["cam"]


def run_end2end_single_batch(model_match, model_egat, model_rf, batch, set_loss, args, device):
    graph, msg = batch
    graph = graph.to(device)
    label_match = graph.ndata["label_match"]["others"]
    label_pose = graph.ndata["label_pos"]["others"]
    label_match_cam = _single_match_cam_target(graph)

    match_out, egat_graph = model_match(graph, msg)
    nodes_retain = get_retain_nodes(egat_graph, mode="flag")
    moving_keep, base_keep = nodes_retain["moving"], nodes_retain["base"]
    rf_out = apply_range_filter(model_rf, egat_graph, args)

    if args.use_CPGO:
        initial_out = model_egat(egat_graph, initial=True)
        pred_tuple = get_pgo_input_multi(egat_graph, initial_out, rf_out, nodes_retain, args)
        moving_pos, base_pose, _ = pgo_multi.run_pgo(
            *pred_tuple,
            fix_weight=args.fix_PGO_weight,
            get_pgo_info=False,
            device=device,
        )
        egat_graph.nodes["moving"].data["feat"][moving_keep] = moving_pos.reshape(-1, 3)
        egat_graph.nodes["base"].data["feat"][base_keep] = base_pose.reshape(-1, 7)

    egat_out = model_egat(egat_graph, initial=False)
    if args.use_RPGO:
        pred_tuple = get_pgo_input_multi(egat_graph, egat_out, rf_out, nodes_retain, args)
        moving_pos, base_pose, _ = pgo_multi.run_pgo(
            *pred_tuple,
            fix_weight=args.fix_PGO_weight,
            get_pgo_info=False,
            device=device,
        )
        egat_out["moving_pos"].append(moving_pos.reshape(-1, 3))
        egat_out["base_pose"].append(base_pose.reshape(-1, 7))
        match_out["pose"] = egat_out["base_pose"][-1]
        loss_type = "match_6dpose"
    else:
        if label_match_cam is None:
            raise ValueError("single end2end without RPGO requires graph.ndata['label_match']['cam'] for scores_3dpos_cov.")
        match_out["pose"] = None
        loss_type = "scores_3dpos_cov"

    target = {"match": label_match, "pose": label_pose, "match_cam": label_match_cam}
    loss = set_loss(match_out, target, supervised_type=loss_type)
    return loss, match_out, target


def run_end2end_compact_batch(model_match, model_egat, model_rf, batch, set_loss, args, device, use_egat_pose_without_rpgo=False):
    graph_seq, msg = batch
    graph_seq = [graph.to(device) for graph in graph_seq]
    label_match, label_pose, label_match_cam = None, None, None
    for graph in graph_seq:
        if label_match is None:
            label_match = graph.ndata["label_match"]["others"]
            label_pose = graph.ndata["label_pos"]["others"]
            label_match_cam = graph.ndata["label_match"]["cam"]
        else:
            label_match = torch.cat((label_match, graph.ndata["label_match"]["others"]), dim=0)
            label_pose = torch.cat((label_pose, graph.ndata["label_pos"]["others"]), dim=0)
            label_match_cam = torch.cat((label_match_cam, graph.ndata["label_match"]["cam"]), dim=0)

    match_out, egat_graph = model_match(graph_seq, msg)
    nodes_retain = get_retain_nodes(egat_graph, mode="flag")
    moving_keep, base_keep = nodes_retain["moving"], nodes_retain["base"]
    rf_out = apply_range_filter(model_rf, egat_graph, args)

    if args.use_CPGO:
        initial_out = model_egat(egat_graph, initial=True)
        pred_tuple = get_pgo_input_multi(egat_graph, initial_out, rf_out, nodes_retain, args)
        moving_pos, base_pose, _ = pgo_multi.run_pgo(
            *pred_tuple,
            fix_weight=args.fix_PGO_weight,
            get_pgo_info=False,
            device=device,
        )
        egat_graph.nodes["moving"].data["feat"][moving_keep] = moving_pos.reshape(-1, 3)
        egat_graph.nodes["base"].data["feat"][base_keep] = base_pose.reshape(-1, 7)

    egat_out = model_egat(egat_graph, initial=False)
    if args.use_RPGO:
        pred_tuple = get_pgo_input_multi(egat_graph, egat_out, rf_out, nodes_retain, args)
        moving_pos, base_pose, _ = pgo_multi.run_pgo(
            *pred_tuple,
            fix_weight=args.fix_PGO_weight,
            get_pgo_info=False,
            device=device,
        )
        egat_out["moving_pos"].append(moving_pos.reshape(-1, 3))
        egat_out["base_pose"].append(base_pose.reshape(-1, 7))
        match_out["pose"] = egat_out["base_pose"][-1]
        loss_type = "match_6dpose_compact_egat"
    elif use_egat_pose_without_rpgo:
        match_out["pose"] = egat_out["base_pose"][-1]
        loss_type = "match_6dpose_compact_egat"
    else:
        match_out["pose"] = None
        loss_type = "scores_3dpos_cov_compact"

    target = {"match": label_match, "pose": label_pose, "match_cam": label_match_cam}
    loss = set_loss(match_out, target, supervised_type=loss_type)
    return loss, match_out, target


def update_compact_recurrent_prior(graph_seq, msg, last_t, preds_queue, args):
    timestamp = _timestamp_from_msg(msg)
    is_consecutive = last_t is not None and abs(timestamp - last_t) <= args.timestamp_thres
    if not preds_queue.empty() and is_consecutive:
        preds_seq = list(preds_queue.queue)
        if len(preds_seq) == args.frame_win:
            del preds_seq[0]
        preds_seq = preds_seq + [preds_seq[-1]]
        if len(preds_seq) < args.frame_win:
            preds_seq = [None] * int(args.frame_win - len(preds_seq)) + preds_seq
        for graph, pred in zip(graph_seq, preds_seq):
            if pred is not None:
                graph.ndata["feat"]["others"][:, :7] = pred
    else:
        print("reset prior")
    return timestamp


def update_single_recurrent_prior(graph, msg, last_t, last_pred, args):
    timestamp = _timestamp_from_msg(msg)
    is_consecutive = last_t is not None and abs(timestamp - last_t) <= args.timestamp_thres
    if last_pred is not None and is_consecutive:
        graph.ndata["feat"]["others"][:, :7] = last_pred.to(graph.device)
    else:
        graph.ndata["feat"]["others"][:, :7] = graph.ndata["label_pos"]["others"]
        print("reset prior")
    return timestamp


def _build_single_objects(args, device):
    dataset = load_graph_dataset(args, args.val_dataset, compact=True)
    dataloader = build_dataloader(args, dataset, shuffle=False)
    model_match, model_egat, model_rf = _build_end2end_backend(
        args,
        device,
        lambda model_args, model_device: MatchSingleEgat(
            model_args,
            model_device,
            return_graph=True,
            record_graph=args.record_graph,
        ),
    )
    return dataloader, model_match, model_egat, model_rf


def _build_compact_objects(args, device):
    dataset = load_sequence_dataset(args, args.val_dataset)
    dataloader = build_dataloader(args, dataset, shuffle=False)
    model_match, model_egat, model_rf = _build_end2end_backend(args, device, MatchCompactEgat)
    return dataloader, model_match, model_egat, model_rf


def _save_recorded_graph_if_requested(model_match, args):
    if not args.record_graph:
        return
    recorder = getattr(model_match, "RF", None)
    if recorder is None:
        raise ValueError("The selected model does not expose a graph recorder.")
    recorder.save_to_csv_ref()


def evaluate_end2end(args):
    _ensure_end2end_args(args)
    device = get_device(args)
    set_loss = losses.SetLoss(args)
    meter = MetricMeter()
    if args.match_mode == "single":
        dataloader, model_match, model_egat, model_rf = _build_single_objects(args, device)
        step_fn = run_end2end_single_batch
        step_extra = {}
    else:
        dataloader, model_match, model_egat, model_rf = _build_compact_objects(args, device)
        step_fn = run_end2end_compact_batch
        step_extra = {"use_egat_pose_without_rpgo": False}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, start=1):
            if args.max_eval_batches > 0 and batch_idx > args.max_eval_batches:
                break
            loss, _, _ = step_fn(model_match, model_egat, model_rf, batch, set_loss, args, device, **step_extra)
            meter.update(loss)
    metrics = meter.compute(rmse_keys=("pos",))
    _save_recorded_graph_if_requested(model_match, args)
    _print_end2end_metrics(metrics, title=f"End2End {args.match_mode} Validation")
    return metrics


def infer_end2end(args):
    _ensure_end2end_args(args)
    device = get_device(args)
    set_loss = losses.SetLoss(args)
    meter = MetricMeter()

    if args.match_mode == "single":
        dataloader, model_match, model_egat, model_rf = _build_single_objects(args, device)
        last_t, last_pred = None, None
        with torch.no_grad():
            for batch_idx, (graph, msg) in enumerate(dataloader, start=1):
                if args.max_eval_batches > 0 and batch_idx > args.max_eval_batches:
                    break
                last_t = update_single_recurrent_prior(graph, msg, last_t, last_pred, args)
                loss, out, target = run_end2end_single_batch(model_match, model_egat, model_rf, (graph, msg), set_loss, args, device)
                meter.update(loss)
                last_pred = out["pose"].reshape(-1, 7) if args.use_RPGO and out.get("pose") is not None else None
                if args.wrt_traj:
                    if out.get("pose") is None:
                        raise ValueError("single end2end trajectory writing requires --use_RPGO true.")
                    write_end2end_traj(out["pose"], target["pose"], _local2map_from_msg(msg, args), last_t, args.model_file, args)
    else:
        dataloader, model_match, model_egat, model_rf = _build_compact_objects(args, device)
        preds_queue = Queue(args.frame_win)
        last_t = None
        with torch.no_grad():
            for batch_idx, (graph_seq, msg) in enumerate(dataloader, start=1):
                if args.max_eval_batches > 0 and batch_idx > args.max_eval_batches:
                    break
                last_t = update_compact_recurrent_prior(graph_seq, msg, last_t, preds_queue, args)
                loss, out, target = run_end2end_compact_batch(
                    model_match,
                    model_egat,
                    model_rf,
                    (graph_seq, msg),
                    set_loss,
                    args,
                    device,
                    use_egat_pose_without_rpgo=True,
                )
                meter.update(loss)
                if preds_queue.full():
                    preds_queue.get()
                preds_queue.put(out["pose"].reshape(-1, 7))
                if args.wrt_traj:
                    label_pose = target["pose"].reshape(args.frame_win, -1, 7)[-1]
                    write_end2end_traj(out["pose"], label_pose, _local2map_from_msg(msg, args), last_t, args.model_file, args)

    metrics = meter.compute(rmse_keys=("pos",))
    _save_recorded_graph_if_requested(model_match, args)
    _print_end2end_metrics(metrics, title=f"End2End {args.match_mode} Inference")
    return metrics


def write_end2end_traj(pred, label, transform, timestamp, path, args):
    os.makedirs(path, exist_ok=True)
    pred_base = pred.cpu().numpy().reshape(args.robot_num, args.robot_num - 1, 7)
    gt_base = label.cpu().numpy().reshape(args.robot_num, args.robot_num - 1, 7)
    for i in range(args.robot_num - 1):
        this_id = ind2id(args.wrt_ref_id, i)
        gt_map = trans_quatpose(gt_base[args.wrt_ref_id, i], transform, w_first=False)
        pred_map = trans_quatpose(pred_base[args.wrt_ref_id, i], transform, w_first=False)
        with open(os.path.join(path, f"gt{this_id}.txt"), "a") as f_gt:
            f_gt.write(" ".join(str(p) for p in [timestamp] + gt_map) + "\n")
        with open(os.path.join(path, f"pred{this_id}.txt"), "a") as f_pred:
            f_pred.write(" ".join(str(p) for p in [timestamp] + pred_map) + "\n")
