# AnyAmber

AnyAmber is a research codebase for learning-based range-aided pose estimation and multi-robot pose graph optimization. The project includes training, evaluation, and inference pipelines for EGAT-based localization, matching networks, range filtering, and end-to-end pose refinement.

## Project Structure

- `args.py`: shared command-line arguments and optional JSON/YAML config loading.
- `train_egat.py`, `train_match.py`, `train_range_filter.py`: training entry points for the main model components.
- `eval.py`, `infer.py`: evaluation and inference entry points.
- `model/`: neural network modules for EGAT, matching, and range filtering.
- `runners/`: task-specific training, evaluation, and inference loops.
- `pgo/`: PyTorch/Theseus pose graph optimization backends.
- `util/`: graph processing, losses, data loading helpers, recording utilities, and geometry helpers.

## Basic Usage

Train an EGAT model:

```bash
python train_egat.py --task egat --mode train
```

Evaluate a model:

```bash
python eval.py --task egat --mode eval --load_pretrained_model true
```

Run inference:

```bash
python infer.py --task egat --mode infer --load_pretrained_model true
```

