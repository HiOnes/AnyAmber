import os
from functools import partial

import torch

import util.losses as losses
from pgo import multi_robot as pgo_multi
from pgo import single_robot as pgo_single
from model.range_filter import RangeFilter
from model.egat import Egat
from runners.common import (
    MetricMeter,
    build_dataloader,
    get_device,
    load_graph_dataset,
    load_state_if_requested,
    print_eval,
    print_train_val,
    save_state,
)
from util.graph_process import get_pgo_input, get_pgo_input_multi, get_retain_nodes


RANGE_EDGE_TYPES = (
    ("moving", "moving2fixed", "fixed"),
    ("fixed", "fixed2moving", "moving"),
    ("moving", "moving2moving", "moving"),
)


def classify_robot_state(args):
    if args.train_RF and not args.use_RF:
        raise ValueError("train_RF requires use_RF to be True")
    if args.robot_num == 1:
        return "SINGLE"
    if args.robot_num > 1:
        return "MULTI"
    if args.robot_num == 0:
        if args.use_CPGO or args.use_RPGO:
            raise ValueError("CPGO and RPGO are not supported when robot_num=0.")
        return "VARYING"
    raise ValueError("robot_num must be >= 0")


def get_input_format(args):
    n_in_dict = {"moving": 3, "fixed": 3, "base": 7, "ref": 7}
    e_in_dict = {
        "moving2fixed": 1,
        "fixed2moving": 1,
        "moving2moving": 1,
        "moving2base": 3,
        "ref2base": 3,
    }
    if args.robot_num == 1 and args.with_bearing:
        e_in_dict["base2fixed"] = 3
    return n_in_dict, e_in_dict


def build_models(args, device):
    n_in_dict, e_in_dict = get_input_format(args)
    model_egat = Egat(
        node_in_dict=n_in_dict,
        edge_in_dict=e_in_dict,
        hid_dim=args.embed_dim,
        gat_heads=args.gat_heads,
        gat_layers=args.gat_layers,
    ).to(device)
    model_rf = RangeFilter(max_num=64, hidden_size=args.embed_dim, device=device).to(device)
    train_rf = args.use_RF and args.train_RF
    for param in model_rf.parameters():
        param.requires_grad = train_rf
    return model_egat, model_rf, train_rf


def build_pgo_adapter(robot_state, args, device):
    if robot_state == "SINGLE":
        pgo_func = partial(
            pgo_single.run_pgo,
            fix_weight=args.fix_PGO_weight,
            get_pgo_info=args.get_PGO_info,
            device=device,
        )
        return pgo_func, get_pgo_input
    if robot_state == "MULTI":
        pgo_func = partial(
            pgo_multi.run_pgo,
            fix_weight=args.fix_PGO_weight,
            get_pgo_info=args.get_PGO_info,
            device=device,
        )
        return pgo_func, get_pgo_input_multi
    return None, None


def apply_range_filter(model_rf, graph, args):
    rf_out = {"filter_range": {}, "range_cov": {}}
    for e_key in RANGE_EDGE_TYPES:
        if e_key not in graph.canonical_etypes:
            continue
        efeat = graph.edata["feat"][e_key]
        if args.use_RF:
            rf = model_rf(
                efeat,
                graph.edata["eid"][e_key],
                use_sensor_embedding=args.use_RF_sensor_embedding,
            )
            rf_out["filter_range"][e_key] = rf["filter_range"]
            rf_out["range_cov"][e_key] = rf["range_cov"]
            graph.edges[e_key].data["feat"] = rf["filter_range"]
        else:
            latest_range = efeat[:, -1].unsqueeze(-1)
            rf_out["filter_range"][e_key] = latest_range
            rf_out["range_cov"][e_key] = torch.ones_like(latest_range)
            graph.edges[e_key].data["feat"] = latest_range
    return rf_out


def egat_step(robot_state, model_egat, model_rf, batch, set_loss, pgo_func, get_pgo_input_func, args, device):
    batched_graph, _ = batch
    graph = batched_graph.to(device)
    nodes_retain = get_retain_nodes(graph, mode="flag")
    moving_keep, base_keep = nodes_retain["moving"], nodes_retain["base"]

    if args.use_RF and not args.train_RF:
        with torch.no_grad():
            rf_out = apply_range_filter(model_rf, graph, args)
    else:
        rf_out = apply_range_filter(model_rf, graph, args)

    if args.use_CPGO:
        initial_out = model_egat(graph, initial=True)
        pred_tuple = get_pgo_input_func(graph, initial_out, rf_out, nodes_retain, args)
        moving_pos, base_pose, _ = pgo_func(*pred_tuple)
        moving_pos, base_pose = moving_pos.reshape(-1, 3), base_pose.reshape(-1, 7)
        if robot_state == "SINGLE":
            moving_pos = moving_pos[moving_keep]
        initial_out["moving_pos"].append(moving_pos)
        initial_out["base_pose"].append(base_pose)
        graph.nodes["moving"].data["feat"][moving_keep] = moving_pos
        graph.nodes["base"].data["feat"][base_keep] = base_pose

    egat_out = model_egat(graph, initial=False)

    if args.use_RPGO:
        pred_tuple = get_pgo_input_func(graph, egat_out, rf_out, nodes_retain, args)
        moving_pos, base_pose, _ = pgo_func(*pred_tuple)
        moving_pos, base_pose = moving_pos.reshape(-1, 3), base_pose.reshape(-1, 7)
        if robot_state == "SINGLE":
            moving_pos = moving_pos[moving_keep]
        egat_out["moving_pos"].append(moving_pos)
        egat_out["base_pose"].append(base_pose)

    target = {
        "moving_pos": graph.ndata["label"]["moving"][moving_keep],
        "base_pose": graph.ndata["label"]["base"][base_keep],
        "multi_tag": graph.ndata["multi_tag"]["base"].bool()[base_keep],
    }
    loss_type = "ref_pgo" if args.use_RPGO else "ref"
    loss = set_loss(egat_out, target, supervised_type=loss_type)
    return loss, egat_out, target


def run_epoch(robot_state, model_egat, model_rf, dataloader, training, train_rf, optimizer, set_loss, pgo_func, get_pgo_input_func, args, device):
    meter = MetricMeter()
    model_egat.train(training)
    model_rf.train(training and train_rf)

    with torch.set_grad_enabled(training):
        for batch_idx, batch in enumerate(dataloader, start=1):
            if not training and args.max_eval_batches > 0 and batch_idx > args.max_eval_batches:
                break
            loss, _, _ = egat_step(
                robot_state,
                model_egat,
                model_rf,
                batch,
                set_loss,
                pgo_func,
                get_pgo_input_func,
                args,
                device,
            )
            if training:
                optimizer.zero_grad()
                loss["total"].backward()
                optimizer.step()
            meter.update(loss)
    return meter.compute(rmse_keys=("pos", "tagpos"))


def build_objects(args, include_train=True):
    device = get_device(args)
    train_dataset = load_graph_dataset(args, args.train_dataset, compact=args.robot_num > 1) if include_train else None
    val_dataset = load_graph_dataset(args, args.val_dataset, compact=args.robot_num > 1)
    robot_state = classify_robot_state(args)
    if train_dataset is not None:
        print(f"Train dataset: {args.train_dataset}, Len: {len(train_dataset)}")
    print(f"Val dataset: {args.val_dataset}, Len: {len(val_dataset)}")
    print(f"Robot state: {robot_state}")

    train_loader = build_dataloader(args, train_dataset, shuffle=args.shuffle) if train_dataset is not None else None
    val_loader = build_dataloader(args, val_dataset, shuffle=False)
    model_egat, model_rf, train_rf = build_models(args, device)
    load_state_if_requested(
        model_egat,
        os.path.join(args.model_file, args.egat_net_checkpoints),
        device,
        args.load_pretrained_model,
    )
    load_state_if_requested(
        model_rf,
        os.path.join(args.model_file, args.rf_net_checkpoints),
        device,
        args.load_pretrained_model and args.use_RF,
    )
    params = list(filter(lambda p: p.requires_grad, model_egat.parameters()))
    params += list(filter(lambda p: p.requires_grad, model_rf.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=5e-4)
    set_loss = losses.SetLoss(args)
    pgo_func, get_pgo_input_func = build_pgo_adapter(robot_state, args, device)
    return {
        "device": device,
        "robot_state": robot_state,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "model_egat": model_egat,
        "model_rf": model_rf,
        "train_rf": train_rf,
        "optimizer": optimizer,
        "set_loss": set_loss,
        "pgo_func": pgo_func,
        "get_pgo_input_func": get_pgo_input_func,
    }


def train(args):
    objects = build_objects(args, include_train=True)
    os.makedirs(args.model_file, exist_ok=True)
    best_val_loss = float("inf")
    keys = ("total", "tagpos", "pos", "rot", "cov")
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            objects["robot_state"],
            objects["model_egat"],
            objects["model_rf"],
            objects["train_loader"],
            True,
            objects["train_rf"],
            objects["optimizer"],
            objects["set_loss"],
            objects["pgo_func"],
            objects["get_pgo_input_func"],
            args,
            objects["device"],
        )
        val_metrics = run_epoch(
            objects["robot_state"],
            objects["model_egat"],
            objects["model_rf"],
            objects["val_loader"],
            False,
            objects["train_rf"],
            None,
            objects["set_loss"],
            objects["pgo_func"],
            objects["get_pgo_input_func"],
            args,
            objects["device"],
        )
        if epoch % args.print_every == 0:
            print_train_val(epoch, train_metrics, val_metrics, keys, title="EGAT")
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            print_train_val(epoch, train_metrics, val_metrics, keys, title="EGAT best")
            save_state(objects["model_egat"], os.path.join(args.model_file, "best_egat.pt"))
            save_state(objects["model_egat"], os.path.join(args.model_file, f"egat-e-{epoch:04d}.pt"))
            if objects["train_rf"]:
                save_state(objects["model_rf"], os.path.join(args.model_file, "best_rf.pt"))
                save_state(objects["model_rf"], os.path.join(args.model_file, f"rf-e-{epoch:04d}.pt"))


def evaluate(args):
    args.mode = "eval"
    objects = build_objects(args, include_train=False)
    metrics = run_epoch(
        objects["robot_state"],
        objects["model_egat"],
        objects["model_rf"],
        objects["val_loader"],
        False,
        objects["train_rf"],
        None,
        objects["set_loss"],
        objects["pgo_func"],
        objects["get_pgo_input_func"],
        args,
        objects["device"],
    )
    print_eval(metrics, ("total", "tagpos", "pos", "rot", "cov"), title="EGAT Validation")
    return metrics
