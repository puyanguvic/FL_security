# Config Parameter Manual

This manual documents `flbench-run` parameters and their defaults, types, and constraints.
You can pass parameters via CLI flags or YAML/JSON configs.

## Config precedence

- Multiple `--config` files are supported; later files override earlier ones.
- CLI flags override config values.

## Aliases (backward compatible)

- `n_clients` -> `num_clients`
- `num_rounds` -> `global_rounds`
- `aggregation_epochs` -> `local_epochs`

Both the old and new CLI flags are accepted.

## Core run parameters

- `algo: str` (required unless provided in config)
  - One of: `fedavg`, `fedprox`, `scaffold`
- `task: str` (required unless provided in config)
  - One of: `cifar10`, `fashionmnist`, `tiny_imagenet`, `har`
- `model: str = "cnn/moderate"`
  - Example: `vgg11`, `alexnet`
- `name: str = null`
  - Optional run/job name override.

## Federated parameters

- `seed: int = 42` (int only)
- `global_rounds: int = 20` (int only, must be > 0)
- `num_clients: int = 10` (int only, must be >= 1)
- `client_fraction: float = 0.5` (float only, 0 < f <= 1)
  - Used to set server `min_clients = ceil(num_clients * client_fraction)`.
  - Note: clients are not subsampled; all clients still receive the task.
- `local_epochs: int = 5` (int only, must be >= 1)
- `batch_size: int = 32` (int only, must be >= 1)

## Optimization

- `optimizer: str = "sgd"` (one of `sgd`, `adam`, `momentum`)
  - `momentum` uses SGD with momentum=0.9
- `lr: float = 0.05`
- `prox_mu: float = 0.0`
  - FedProx mu coefficient (only for `fedprox`)
- `num_workers: int = 2`

## Data / split

- `alpha: float = 0.5`
  - Dirichlet alpha for non-iid splits (must be > 0)
- `data_root: str = null`
  - Task-specific dataset root path

## Device

- `device: str = "cuda:0"`
  - `cpu` or `cuda:{idx}`
  - If CUDA is requested but unavailable, runtime falls back to CPU with a warning.

## Malicious client selection

- `n_malicious: int = 0`
  - Number of malicious clients
- `malicious_mode: str = "random"`
  - `first` or `random`
- `malicious_seed: int = null`
  - Seed for random malicious selection (falls back to `seed` if unset)

## Attacks (client update + server-side byzantine)

- `attack: str = "none"`
  - One of: `none`, `scale`, `sign_flip`, `gaussian`, `pgd_minmax`, `lie`, `fang`, `sme`, `backdoor`, `minmax`, `minsum`, `byz_gaussian`
  - Update attacks (`scale`, `sign_flip`, `gaussian`, `pgd_minmax`) run on malicious clients before sending updates.
  - Byzantine attacks (`lie`, `fang`, `sme`, `backdoor`, `minmax`, `minsum`, `byz_gaussian`) run on the server over collected updates and require `n_malicious > 0`.
- `attack_kv: [key=value] = []`
  - Repeatable key/value overrides for attack parameters
- `attack_config: str = null`
  - Path to YAML/JSON file providing attack params
- `attack_scale: float = 1.0`
  - For `scale`
- `attack_noise_std: float = 0.0`
  - For `gaussian`
- `attack_pgd_steps: int = 1`
- `attack_pgd_step_size: float = 0.1`
- `attack_pgd_eps: float = 0.0`
- `attack_pgd_eps_factor: float = 1.0`
- `attack_pgd_max_batches: int = 1`
- `attack_pgd_init: str = "zero"`
  - `zero`, `local`, or `sign`
- `attack_seed: int = null`
  - Optional seed for stochastic attacks

### Byzantine attack params (use `attack_kv` / `attack_config`)

- `learning_rate` / `lr` / `attack_lr`: for `sme`, `minmax`, `minsum` (defaults to `lr`)
- `surrogate_scale`: for `sme` (default `1.0`)
- `attacker_ability`: for `sme`, `backdoor`, `minmax` (`Full` or `Part`)
- `critical_layer_names` / `layers`: for `backdoor` (list or comma-separated string)
- `poison_scale`: for `backdoor` (default `1.0`)
- `attack_scale` / `scale`: for `byz_gaussian` (default `attack_scale`)

## Defenses (server aggregation)

- `defense: str = "none"`
  - One of: `none`, `mean`, `multikrum`, `wbc`, `fgnv`, `fldetector`, `beta_reputation`
  - Availability depends on optional dependencies.
- `defense_kv: [key=value] = []`
  - Repeatable key/value overrides for defense parameters
- `defense_config: str = null`
  - Path to YAML/JSON file providing defense params
- `defense_trim_ratio: float = 0.0`
- `defense_clip_norm: float = 0.0`
- `defense_krum_f: int = -1`
- `defense_krum_m: int = 1`

## Tracking and output

- `tracking: str = "tensorboard"`
  - `tensorboard` or `none`
- `results_dir: str = "experiments/runs"`
  - Destination for copied run artifacts
- `sim_workspace_root: str = "experiments/nvflare/simulation"`
  - NVFLARE simulation workspace root
- `resume: bool = false`
  - Resume from existing simulation workspace if present

## Unified config example

```yaml
algo: fedavg
task: cifar10
model: vgg11
num_clients: 32
global_rounds: 20
client_fraction: 0.5
local_epochs: 4
batch_size: 32
seed: 42

attack:
  name: sign_flip

n_malicious: 8
```
