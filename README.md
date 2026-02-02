# flbench

A research-ready **NVFLARE experimentation lab** with a clean separation between:

- **Algorithms** (FedAvg/FedProx/SCAFFOLD/…)
- **Tasks** (image + sensor datasets)
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
  utils/

src/tasks/
  cifar10/
  fashionmnist/
  tiny_imagenet/
  har/

src/models/
  alexnet.py
  moderate_cnn.py
  vgg11.py
```

## Included modules

- `algo`: `fedavg`, `fedprox`, `scaffold`
- `task`: `cifar10`, `fashionmnist`, `tiny_imagenet`, `har`
- `model`: `cnn/moderate` (plus `vgg11`/`alexnet` in `models/`)

## Quickstart (uv)

```bash
uv venv
uv pip install -e .

# Run FedAvg on CIFAR-10 (simulation)
flbench-run --algo fedavg --task cifar10 --model cnn/moderate --num_clients 10 --global_rounds 20 --alpha 0.5

# Run FedAvg on FashionMNIST (simulation)
flbench-run --algo fedavg --task fashionmnist --model cnn/moderate --num_clients 10 --global_rounds 20 --alpha 0.5
```

## Experiments

Experiment 1 (no attack):

```bash
flbench-run --algo fedavg --task cifar10 --model vgg11 \
  --attack none --global_rounds 20 --num_clients 8 --client_fraction 0.5 \
  --n_malicious 0 --local_epochs 4 --batch_size 8 --device cuda:0
```

Experiment 2 (sign-flip attack):

```bash
flbench-run --algo fedavg --task cifar10 --model vgg11 \
  --attack sign_flip --global_rounds 20 --num_clients 8 --client_fraction 0.5 \
  --n_malicious 2 --local_epochs 4 --batch_size 8 --device cuda:0
```

Experiment 3 (attack + defense):

```bash
flbench-run --algo fedavg --task cifar10 --model vgg11 \
  --attack sign_flip --defense multikrum --defense_krum_f 2 --defense_krum_m 1 \
  --global_rounds 20 --num_clients 8 --client_fraction 0.5 \
  --n_malicious 2 --local_epochs 4 --batch_size 8 --device cuda:0
```

## Config files (unified)

You can provide one or more YAML/JSON configs via `--config`. Later files override earlier ones, and CLI flags
override config values.

Compose small configs:

```bash
flbench-run --config configs/default.yaml --config configs/algo_fedavg.yaml --config configs/task_cifar10.yaml
```

Or use a single unified file:

```yaml
# configs/run.yaml
algo: fedavg
task: cifar10
model: cnn/moderate
num_clients: 10
global_rounds: 20
alpha: 0.5

attack:
  name: pgd_minmax
  steps: 3
  step_size: 0.1

defense:
  name: multikrum
  params:
    f: 2
    m: 1
```

`attack`/`defense` blocks accept either `params` (inline dict) or `config` (path to a YAML/JSON file), and you can
still use top-level `attack_config`/`defense_config` if you prefer.

Full parameter manual: `CONFIG_MANUAL.md`.

## Common parameters

Defaults and constraints:

- `seed: int = 42` (int only)
- `global_rounds: int = 20` (int only, must be > 0)
- `num_clients: int = 10` (int only, must be >= 1)
- `client_fraction: float = 0.5` (float only, 0 < f <= 1)
- `local_epochs: int = 5` (int only, must be >= 1)
- `batch_size: int = 32` (int only, must be >= 1)
- `optimizer: str = "sgd"` (one of `sgd`, `adam`, `momentum`)
- `device: str = "cuda:0"` (`cpu` or `cuda:{idx}`)

Backward-compatible aliases are accepted in configs/CLI:
`n_clients` -> `num_clients`, `num_rounds` -> `global_rounds`, `aggregation_epochs` -> `local_epochs`.

## Where outputs go

All experiment artifacts live under `./experiments/` by default:

- Data split indices: `./experiments/splits/...`
- NVFLARE simulation workspace: `./experiments/nvflare/simulation` (override with `--sim_workspace_root`)
- Copied run results: `./experiments/runs/<job_name>/` (override with `--results_dir`)
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
Create `src/tasks/<domain>/<task_name>/` with:
- `dataset.py` (dataset + loaders + task-specific transforms)
- `split.py` (optional non-iid split)
- `task.py` (build_model + split_and_save + dataset factories)

Register it in `src/flbench/core/registry.py`.

### Add a new attack/defense
- Attacks: `src/flbench/attacks/` (register in `attacks/registry.py`)
- Defenses: `src/flbench/defenses/` (register in `defenses/registry.py`)
