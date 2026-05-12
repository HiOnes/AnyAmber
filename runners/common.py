import math
import os
from typing import Iterable, Optional, Sequence

import dgl.data
import torch
from dgl.dataloading import GraphDataLoader

import util.data_pro as dp


def get_device(args):
    requested = getattr(args, "device", "cuda")
    if requested == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if isinstance(requested, str) and requested.isdigit():
        return torch.device(f"cuda:{requested}")
    if isinstance(requested, str) and requested.startswith("cuda"):
        return torch.device(requested)
    return torch.device("cuda")


def load_graph_dataset(args, path: str, compact: Optional[bool] = None):
    if compact is None:
        compact = getattr(args, "robot_num", 0) > 1
    if compact:
        return dp.ComPactedCSVDataset(args.robot_num, path)
    return dgl.data.CSVDataset(path)


def load_sequence_dataset(args, path: str):
    dataset = load_graph_dataset(args, path)
    sequences = dp.create_continues_sequences(dataset, args.frame_win, args.timestamp_thres)
    return dp.TimeSeriesDataset(sequences)


def build_dataloader(args, dataset, shuffle: Optional[bool] = None, drop_last: bool = True):
    if shuffle is None:
        shuffle = args.shuffle
    return GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def load_state_if_requested(model, checkpoint_path: str, device, requested: bool):
    if not requested:
        return False
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded model from {checkpoint_path}")
    return True


def save_state(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


class MetricMeter:
    def __init__(self):
        self.totals = {}
        self.count = 0

    def update(self, loss):
        self.count += 1
        for key, value in loss.items():
            if hasattr(value, "detach"):
                value = value.detach().item()
            self.totals[key] = self.totals.get(key, 0.0) + float(value)

    def compute(self, rmse_keys: Sequence[str] = ()):
        if self.count <= 0:
            raise ValueError("MetricMeter has no batches.")
        metrics = {key: value / self.count for key, value in self.totals.items()}
        for key in rmse_keys:
            if key in metrics:
                metrics[key] = math.sqrt(metrics[key])
        return metrics


def run_epoch(model, dataloader, step_fn, args, device, optimizer=None, rmse_keys: Sequence[str] = ()):
    training = optimizer is not None
    model.train(training)
    meter = MetricMeter()
    with torch.set_grad_enabled(training):
        for batch_idx, batch in enumerate(dataloader, start=1):
            if not training and getattr(args, "max_eval_batches", 0) > 0 and batch_idx > args.max_eval_batches:
                break
            loss = step_fn(model, batch, args, device)
            if training:
                optimizer.zero_grad()
                loss["total"].backward()
                optimizer.step()
            meter.update(loss)
    return meter.compute(rmse_keys=rmse_keys)


def print_train_val(epoch: int, train_metrics, val_metrics, keys: Iterable[str], title: str = ""):
    prefix = f"Epoch {epoch}"
    if title:
        prefix = f"{prefix} | {title}"
    train = " | ".join(f"{key} {train_metrics.get(key, 0.0):.4f}" for key in keys)
    val = " | ".join(f"{key} {val_metrics.get(key, 0.0):.4f}" for key in keys)
    print(f"{prefix} | Train-- {train} || Validation-- {val}")


def print_eval(metrics, keys: Iterable[str], title: str = "Validation"):
    body = " | ".join(f"{key} {metrics.get(key, 0.0):.4f}" for key in keys)
    print(f"***{title}-- {body}***")
