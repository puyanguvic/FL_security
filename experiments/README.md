# Experiments

- `runs/` stores copied NVFLARE simulation outputs (default for `flbench-run`).
- `splits/` stores dataset split indices.
- `nvflare/simulation/` stores the NVFLARE simulation workspace.
- Each run folder contains NVFLARE artifacts, TensorBoard events, and exported metrics.

Export metrics:

```bash
python scripts/export_metrics.py experiments/runs/<job_name>
```
