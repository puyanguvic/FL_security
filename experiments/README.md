# Experiments

- `runs/` stores copied NVFLARE simulation outputs (default for `flbench-run`).
- Each run folder contains NVFLARE artifacts, TensorBoard events, and exported metrics.

Export metrics:

```bash
python scripts/export_metrics.py experiments/runs/<job_name>
```
