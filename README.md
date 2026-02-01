# flbench

A research-ready **NVFLARE experimentation lab** with a clean separation between:

- **Algorithms** (FedAvg/FedProx/SCAFFOLD/…)
- **Tasks** (vision + sensor datasets)
- **Models** (reusable backbones)
- **Attacks / Defenses** (client update attacks and server aggregation defenses)

## Layout

```
configs/
  default.yaml
  algo_fedavg.yaml
  task_cifar10.yaml
  attack_none.yaml
  defense_none.yaml

scripts/
  run_sim.sh
  export_metrics.py

experiments/
  runs/              # gitignored
  README.md

src/flbench/
  cli/               # flbench-run entrypoint
  core/              # abstractions & lifecycle
  algorithms/
  attacks/
  defenses/
  tasks/
  models/
  utils/
```

## Included modules

- `algo`: `fedavg`, `fedprox`, `scaffold`
- `task`: `vision/cifar10`, `vision/fashionmnist`, `vision/tiny_imagenet`, `sensor/har`
- `model`: `cnn/moderate` (plus vision backbones in `models/vision.py`)

## Quickstart (uv)

```bash
uv venv
uv pip install -e .

# Run FedAvg on CIFAR-10 (simulation)
flbench-run --algo fedavg --task vision/cifar10 --model cnn/moderate --n_clients 8 --num_rounds 20 --alpha 0.5

# Run FedAvg on FashionMNIST (simulation)
flbench-run --algo fedavg --task vision/fashionmnist --model cnn/moderate --n_clients 8 --num_rounds 20 --alpha 0.5
```

## Where outputs go

- Data split indices: `/tmp/flbench_splits/...`
- NVFLARE run result path is printed at the end and copied to `./experiments/runs/<job_name>/` (override with `--results_dir`)
- TensorBoard tracking is enabled by default (via recipe tracking)

Export metrics from a run directory:

```bash
python scripts/export_metrics.py experiments/runs/<job_name>
```

## Extend

### Add a new algorithm
Create `src/flbench/algorithms/<algo_name>/` with:
- `client.py` (client-side local training loop)
- `server.py` (recipe + orchestration)

Register it in `src/flbench/core/registry.py`.

### Add a new task
Create `src/flbench/tasks/<domain>/<task_name>/` with:
- `dataset.py` (dataset + loaders + task-specific transforms)
- `split.py` (optional non-iid split)
- `task.py` (build_model + split_and_save + dataset factories)

Register it in `src/flbench/core/registry.py`.

### Add a new attack/defense
- Attacks: `src/flbench/attacks/` (register in `attacks/registry.py`)
- Defenses: `src/flbench/defenses/` (register in `defenses/registry.py`)
