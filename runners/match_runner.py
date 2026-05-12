import os
import warnings

import torch

import util.data_pro as dp
import util.losses as losses
from pgo import multi_robot_single_tag as pgo
from runners.common import (
    MetricMeter,
    build_dataloader,
    get_device,
    load_state_if_requested,
    print_eval,
    print_train_val,
    save_state,
)
from util.utils import get_T_from_quatpose, ind2id, trans_pos, trans_quatpose


MATCH_KEYS = {
    "single": ("total", "match", "precision", "recall", "pos", "rot", "cov"),
    "handcraft": ("total", "match", "precision", "recall", "pos", "rot", "cov"),
    "compact": ("total", "match", "precision", "recall", "precision_last", "recall_last", "pos", "rot", "cov"),
    "split": ("total", "match", "precision", "recall", "pos", "rot", "cov"),
    "seq": (
        "total",
        "match",
        "precision_others2cam",
        "recall_others2cam",
        "precision_cam2cam",
        "recall_cam2cam",
        "pos",
        "rot",
        "cov",
    ),
}


def _match_mode(args):
    return getattr(args, "match_mode", "compact")


def _check_match_args(args):
    mode = _match_mode(args)
    if args.robot_num <= 1:
        raise ValueError("match task expects robot_num > 1 fixed multi-robot data.")
    if mode in ("split", "seq") and args.use_RPGO:
        raise ValueError(f"{mode} mode does not support RPGO/PGO yet.")
    return mode


def _build_match_model(args, device):
    mode = _check_match_args(args)
    if mode == "single":
        from model.match_single import MatchSingle as MatchModel
    elif mode == "handcraft":
        from model.match_handcrafted import MatchHandcrafted as MatchModel
    elif mode == "compact":
        from model.match_compact import MatchCompact as MatchModel
    elif mode == "split":
        from model.match_split import MatchSplit as MatchModel
    elif mode == "seq":
        from model.match_sequence import MatchSequence as MatchModel
    else:
        raise ValueError(f"Unknown match_mode: {mode}")
    return MatchModel(args, device).to(device)


def _load_match_dataset(args, path):
    mode = _check_match_args(args)
    dataset = dp.ComPactedCSVDataset(args.robot_num, path)
    if mode in ("single", "handcraft"):
        return dataset
    if mode == "seq":
        sequences = dp.create_continues_sequences_cam_match_reversed(dataset, args)
    else:
        sequences = dp.create_continues_sequences(dataset, args.frame_win, args.timestamp_thres)
    return dp.TimeSeriesDataset(sequences)


def _collect_single_targets(graph):
    target = {
        "match": graph.ndata["label_match"]["others"],
        "pose": graph.ndata["label_pos"]["others"],
        "match_cam": graph.ndata["label_match"]["cam"],
    }
    prior_pose = graph.ndata["feat"]["others"][:, :7]
    return target, prior_pose


def _collect_basic_targets(graph_seq):
    label_match, label_pose, label_match_cam, prior_pose = None, None, None, None
    for graph in graph_seq:
        if label_match is None:
            label_match = graph.ndata["label_match"]["others"]
            label_pose = graph.ndata["label_pos"]["others"]
            label_match_cam = graph.ndata["label_match"]["cam"]
            prior_pose = graph.ndata["feat"]["others"][:, :7]
        else:
            label_match = torch.cat((label_match, graph.ndata["label_match"]["others"]), dim=0)
            label_pose = torch.cat((label_pose, graph.ndata["label_pos"]["others"]), dim=0)
            label_match_cam = torch.cat((label_match_cam, graph.ndata["label_match"]["cam"]), dim=0)
            prior_pose = torch.cat((prior_pose, graph.ndata["feat"]["others"][:, :7]), dim=0)
    target = {"match": label_match, "pose": label_pose, "match_cam": label_match_cam}
    return target, prior_pose


def _collect_seq_targets(graph_seq):
    target, prior_pose = _collect_basic_targets(graph_seq)
    label_match_src2des, label_match_des2src = None, None
    for graph in graph_seq:
        if label_match_src2des is None:
            label_match_src2des = graph.ndata["label_src2des"]["cam"]
            label_match_des2src = graph.ndata["label_des2src"]["cam"]
        else:
            label_match_src2des = torch.cat((label_match_src2des, graph.ndata["label_src2des"]["cam"]), dim=0)
            label_match_des2src = torch.cat((label_match_des2src, graph.ndata["label_des2src"]["cam"]), dim=0)
    target["match_src2des"] = label_match_src2des
    target["match_des2src"] = label_match_des2src
    return target, prior_pose


def match_step(model, batch, set_loss, args, device):
    loss, _, _ = match_forward(model, batch, set_loss, args, device)
    return loss


def match_forward(model, batch, set_loss, args, device):
    mode = _check_match_args(args)

    if mode in ("single", "handcraft"):
        graph, msg_dict = batch
        graph = graph.to(device)
        target, prior_pose = _collect_single_targets(graph)
        out = model(graph, msg_dict)
        if args.use_RPGO:
            out["pose"] = pgo.run_pgo(out["pos"], out["cov"], prior_pose, robot_num=args.robot_num, device=device)
            loss_type = "match_6dpose"
        elif mode == "handcraft":
            out["pose"] = None
            loss_type = "handcrafted"
        else:
            out["pose"] = None
            loss_type = "scores_3dpos_cov"
    else:
        graph_seq, msg_dict = batch
        graph_seq = [graph.to(device) for graph in graph_seq]
        if mode == "seq":
            target, prior_pose = _collect_seq_targets(graph_seq)
            out = model(graph_seq)
            out["pose"] = None
            loss_type = "scores_3dpos_cov_seq"
        elif mode == "split":
            target, prior_pose = _collect_basic_targets(graph_seq)
            out = model(graph_seq)
            out["pose"] = None
            loss_type = "scores_3dpos_cov"
        else:
            target, prior_pose = _collect_basic_targets(graph_seq)
            out = model(graph_seq, msg_dict)
            if args.use_RPGO:
                out["pose"] = pgo.run_pgo(out["pos"], out["cov"], prior_pose, robot_num=args.robot_num, device=device)
                loss_type = "match_6dpose_compact"
            else:
                out["pose"] = None
                loss_type = "scores_3dpos_cov_compact"

    loss = set_loss(out, target, supervised_type=loss_type)
    return loss, out, target


def run_match_epoch(model, dataloader, set_loss, args, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    meter = MetricMeter()
    with torch.set_grad_enabled(training):
        for batch_idx, batch in enumerate(dataloader, start=1):
            if not training and args.max_eval_batches > 0 and batch_idx > args.max_eval_batches:
                break
            loss = match_step(model, batch, set_loss, args, device)
            if training:
                optimizer.zero_grad()
                loss["total"].backward()
                optimizer.step()
            meter.update(loss)
    return meter.compute(rmse_keys=("pos",))


def build_objects(args, include_train=True):
    device = get_device(args)
    train_dataset = _load_match_dataset(args, args.train_dataset) if include_train else None
    val_dataset = _load_match_dataset(args, args.val_dataset)
    mode = _match_mode(args)
    if train_dataset is not None:
        print(f"Train dataset: {args.train_dataset}, Len: {len(train_dataset)}")
    print(f"Val dataset: {args.val_dataset}, Len: {len(val_dataset)}")
    print(f"Match mode: {mode}")

    train_loader = build_dataloader(args, train_dataset, shuffle=args.shuffle) if train_dataset is not None else None
    val_loader = build_dataloader(args, val_dataset, shuffle=False)
    model = _build_match_model(args, device)
    load_state_if_requested(
        model,
        os.path.join(args.model_file, args.match_net_checkpoints),
        device,
        args.load_pretrained_model,
    )
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
    set_loss = losses.SetLoss(args)
    return {
        "device": device,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "model": model,
        "optimizer": optimizer,
        "set_loss": set_loss,
        "keys": MATCH_KEYS[mode],
    }


def train(args):
    objects = build_objects(args, include_train=True)
    os.makedirs(args.model_file, exist_ok=True)
    best_val_loss = float("inf")
    keys = objects["keys"]
    for epoch in range(args.epochs):
        train_metrics = run_match_epoch(
            objects["model"],
            objects["train_loader"],
            objects["set_loss"],
            args,
            objects["device"],
            optimizer=objects["optimizer"],
        )
        val_metrics = run_match_epoch(
            objects["model"],
            objects["val_loader"],
            objects["set_loss"],
            args,
            objects["device"],
        )
        if epoch % args.print_every == 0:
            print_train_val(epoch, train_metrics, val_metrics, keys, title=f"MatchNet {_match_mode(args)}")
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            print_train_val(epoch, train_metrics, val_metrics, keys, title=f"MatchNet {_match_mode(args)} best")
            save_state(objects["model"], os.path.join(args.model_file, "best_match.pt"))
            save_state(objects["model"], os.path.join(args.model_file, f"match-e-{epoch:04d}.pt"))


def evaluate(args):
    objects = build_objects(args, include_train=False)
    metrics = run_match_epoch(objects["model"], objects["val_loader"], objects["set_loss"], args, objects["device"])
    print_eval(metrics, objects["keys"], title=f"MatchNet {_match_mode(args)} Validation")
    return metrics


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


def _latest_label_pose(target, args):
    if _match_mode(args) in ("single", "handcraft"):
        return target["pose"]
    return target["pose"].reshape(args.frame_win, -1, 7)[-1]


def _latest_pred_pose(out, args, device):
    mode = _match_mode(args)
    if mode in ("single", "handcraft"):
        if out.get("pose") is None:
            raise ValueError(f"{mode} match trajectory writing requires PGO pose; enable use_RPGO.")
        return out["pose"].reshape(-1, 7), False
    if mode == "compact":
        if out.get("pose") is None:
            raise ValueError("compact match trajectory writing requires use_RPGO to produce 6D pose.")
        return out["pose"].reshape(args.frame_win, -1, 7)[-1], False

    if mode == "split":
        pred_pos = out["pos"].reshape(args.frame_win, -1, args.robot_num - 1, 3)[-1].reshape(-1, 3)
    elif mode == "seq":
        pred_pos = out["pos"].reshape(-1, 3)
    else:
        raise ValueError(f"Unknown match_mode: {mode}")

    warnings.warn(
        f"{mode} match mode writes pseudo poses for wrt_traj: position comes from MatchNet out['pos'] "
        "and quaternion is fixed to [0, 0, 0, 1].",
        RuntimeWarning,
        stacklevel=2,
    )
    identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=pred_pos.dtype, device=device)
    pred_pose = torch.cat([pred_pos, identity_quat.expand(pred_pos.shape[0], 4)], dim=-1)
    return pred_pose, True


def write_match_traj(pred_pose, label_pose, transform, timestamp, path, args, pseudo_pose=False):
    os.makedirs(path, exist_ok=True)
    pred_base = pred_pose.cpu().numpy().reshape(args.robot_num, args.robot_num - 1, 7)
    gt_base = label_pose.cpu().numpy().reshape(args.robot_num, args.robot_num - 1, 7)
    for i in range(args.robot_num - 1):
        this_id = ind2id(args.wrt_ref_id, i)
        gt_map = trans_quatpose(gt_base[args.wrt_ref_id, i], transform, w_first=False)
        if pseudo_pose:
            pred_map = trans_pos(pred_base[args.wrt_ref_id, i, :3], transform) + [0, 0, 0, 1]
        else:
            pred_map = trans_quatpose(pred_base[args.wrt_ref_id, i], transform, w_first=False)
        with open(os.path.join(path, f"gt{this_id}.txt"), "a") as f_gt:
            f_gt.write(" ".join(str(p) for p in [timestamp] + gt_map) + "\n")
        with open(os.path.join(path, f"pred{this_id}.txt"), "a") as f_pred:
            f_pred.write(" ".join(str(p) for p in [timestamp] + pred_map) + "\n")


def infer(args):
    if args.wrt_traj and args.batch_size != 1:
        raise ValueError("match trajectory writing expects batch_size=1.")
    if _match_mode(args) in ("single", "handcraft") and args.wrt_traj and not args.use_RPGO:
        raise ValueError(f"{_match_mode(args)} match trajectory writing requires PGO pose; enable use_RPGO.")
    objects = build_objects(args, include_train=False)
    model = objects["model"]
    device = objects["device"]
    model.load_state_dict(torch.load(os.path.join(args.model_file, args.match_net_checkpoints), map_location=device))
    model.eval()
    meter = MetricMeter()
    warned_pseudo_pose = False
    last_t, last_pred = None, None
    with torch.no_grad():
        for batch_idx, batch in enumerate(objects["val_loader"], start=1):
            if args.max_eval_batches > 0 and batch_idx > args.max_eval_batches:
                break
            if _match_mode(args) in ("single", "handcraft"):
                last_t = update_single_prior(batch, last_t, last_pred, args)
            loss, out, target = match_forward(model, batch, objects["set_loss"], args, device)
            meter.update(loss)
            if _match_mode(args) in ("single", "handcraft") and out.get("pose") is not None:
                last_pred = out["pose"].reshape(-1, 7)
            if args.wrt_traj:
                if _match_mode(args) in ("split", "seq") and warned_pseudo_pose:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        pred_pose, pseudo_pose = _latest_pred_pose(out, args, device)
                else:
                    pred_pose, pseudo_pose = _latest_pred_pose(out, args, device)
                    warned_pseudo_pose = warned_pseudo_pose or pseudo_pose
                _, msg = batch
                write_match_traj(
                    pred_pose,
                    _latest_label_pose(target, args),
                    _local2map_from_msg(msg, args),
                    _timestamp_from_msg(msg),
                    args.model_file,
                    args,
                    pseudo_pose=pseudo_pose,
                )
    metrics = meter.compute(rmse_keys=("pos",))
    print_eval(metrics, objects["keys"], title=f"MatchNet {_match_mode(args)} Inference")
    return metrics


def update_single_prior(batch, last_t, last_pred, args):
    graph, msg = batch
    timestamp = _timestamp_from_msg(msg)
    is_consecutive = last_t is not None and abs(timestamp - last_t) <= args.timestamp_thres
    if last_pred is not None and is_consecutive:
        graph.ndata["feat"]["others"][:, :7] = last_pred.to(graph.device)
    else:
        graph.ndata["feat"]["others"][:, :7] = graph.ndata["label_pos"]["others"]
        print("reset prior")
    return timestamp
