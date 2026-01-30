# nvflare-lab

A research-ready **NVFLARE experimentation lab** with a clean separation between:

- **Algorithms** (e.g., FedAvg/FedProx/SCAFFOLD/…)
- **Tasks** (e.g., CIFAR-10/MNIST/IMDB/tabular/…)
- **Models** (reusable backbones)

This repo currently includes:
- `algo=fedavg`
- `task=vision/cifar10`
- `task=vision/fashionmnist`
- `model=cnn/moderate`

## Quickstart (uv)

```bash
uv venv
uv pip install -e .

# Run FedAvg on CIFAR-10 (simulation)
flbench-run --algo fedavg --task vision/cifar10 --model cnn/moderate --n_clients 8 --num_rounds 20 --alpha 0.5

# Run FedAvg on FashionMNIST (simulation)
flbench-run --algo fedavg --task vision/fashionmnist --model cnn/moderate --n_clients 8 --num_rounds 20 --alpha 0.5
```

### Where outputs go
- Data split indices: `/tmp/flbench_splits/...`
- NVFLARE run result path is printed at the end and copied to `./results/<job_name>/` (override with `--results_dir`)
- Tensorboard tracking is enabled by default (via recipe tracking)

## Extend

### Add a new algorithm
Create a folder under `src/flbench/algorithms/<algo_name>/` with:
- `job.py` (build recipe + execute)
- `client.py` (client-side local training loop)

Register it in `src/flbench/core/registry.py`.

### Add a new task
Create a folder under `src/flbench/tasks/<domain>/<task_name>/` with:
- `dataset.py` (dataset + loaders + task-specific augmentations)
- `split.py` (optional non-iid split)
- `task.py` (task wrapper: `build_model`, `split_and_save`, exports dataset factories)

Register it in `src/flbench/core/registry.py`.
