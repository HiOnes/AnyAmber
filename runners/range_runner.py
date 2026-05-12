import os

import torch

import util.losses as losses
from model.range_filter import RangeFilter
from runners.common import (
    build_dataloader,
    get_device,
    load_graph_dataset,
    load_state_if_requested,
    print_eval,
    print_train_val,
    run_epoch,
    save_state,
)


RANGE_EDGE_TYPE = ("moving", "moving2fixed", "fixed")


def range_step(model, batch, args, device):
    graph, _ = batch
    graph = graph.to(device)
    label_range = graph.edata["label"][RANGE_EDGE_TYPE]
    out = model(
        graph.edata["feat"][RANGE_EDGE_TYPE],
        graph.edata["eid"][RANGE_EDGE_TYPE],
        use_sensor_embedding=args.use_RF_sensor_embedding,
    )
    target = {"range": label_range}
    return losses.SetLoss(args)(out, target, supervised_type="range")


def build_objects(args, include_train=True):
    device = get_device(args)
    train_dataset = load_graph_dataset(args, args.train_dataset, compact=args.robot_num > 1) if include_train else None
    val_dataset = load_graph_dataset(args, args.val_dataset, compact=args.robot_num > 1)
    if train_dataset is not None:
        print(f"Train dataset: {args.train_dataset}, Len: {len(train_dataset)}")
    print(f"Val dataset: {args.val_dataset}, Len: {len(val_dataset)}")

    train_loader = build_dataloader(args, train_dataset, shuffle=args.shuffle) if train_dataset is not None else None
    val_loader = build_dataloader(args, val_dataset, shuffle=False)
    model = RangeFilter(max_num=64, hidden_size=args.embed_dim, device=device).to(device)
    load_state_if_requested(
        model,
        os.path.join(args.model_file, args.rf_net_checkpoints),
        device,
        args.load_pretrained_model,
    )
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
    return {"device": device, "train_loader": train_loader, "val_loader": val_loader, "model": model, "optimizer": optimizer}


def train(args):
    objects = build_objects(args, include_train=True)
    os.makedirs(args.model_file, exist_ok=True)
    best_val_loss = float("inf")
    keys = ("total", "rot", "cov")
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            objects["model"],
            objects["train_loader"],
            range_step,
            args,
            objects["device"],
            optimizer=objects["optimizer"],
        )
        val_metrics = run_epoch(
            objects["model"],
            objects["val_loader"],
            range_step,
            args,
            objects["device"],
        )
        if epoch % args.print_every == 0:
            print_train_val(epoch, train_metrics, val_metrics, keys, title="RangeFilter")
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            print_train_val(epoch, train_metrics, val_metrics, keys, title="RangeFilter best")
            save_state(objects["model"], os.path.join(args.model_file, "best_rf.pt"))
            save_state(objects["model"], os.path.join(args.model_file, f"rf-e-{epoch:04d}.pt"))


def evaluate(args):
    objects = build_objects(args, include_train=False)
    metrics = run_epoch(objects["model"], objects["val_loader"], range_step, args, objects["device"])
    print_eval(metrics, ("total", "rot", "cov"), title="RangeFilter Validation")
    return metrics
