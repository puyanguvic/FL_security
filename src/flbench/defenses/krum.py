from __future__ import annotations

import torch

from flbench.defenses.base import Defense, DefenseContext, unweighted_mean, weighted_mean
from flbench.defenses.registry import register_defense


def _flatten_update(update):
    flat = []
    for _, v in update.items():
        flat.append(v.float().reshape(-1))
    if not flat:
        return torch.zeros(1)
    return torch.cat(flat, dim=0)


def _krum_select(vectors, f: int, m: int):
    n = len(vectors)
    if n == 1:
        return [0]
    if n < 2 * f + 3:
        return list(range(n))

    dists = torch.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = torch.norm(vectors[i] - vectors[j]).pow(2)
            dists[i, j] = dist
            dists[j, i] = dist

    scores = []
    k = n - f - 2
    for i in range(n):
        vals, _ = torch.sort(dists[i])
        score = vals[1 : k + 1].sum() if k > 0 else vals[1:].sum()
        scores.append((float(score.item()), i))
    scores.sort(key=lambda x: x[0])
    m = max(1, min(m, n))
    return [idx for _, idx in scores[:m]]


@register_defense("krum")
class KrumDefense(Defense):
    def __init__(self, f: int | None = None, m: int = 1, n_malicious: int = 0):
        self.f = f
        self.m = int(m)
        self.n_malicious = int(n_malicious)

    def apply(self, updates, ctx: DefenseContext):
        if not updates:
            return {}, {"defense": "krum", "skipped": 1}
        f = self.f if self.f is not None and self.f >= 0 else self.n_malicious
        f = max(0, int(f))
        vectors = [_flatten_update(u) for u in updates]
        selected = _krum_select(vectors, f=f, m=self.m)
        if len(selected) == len(updates):
            return weighted_mean(updates, ctx.weights), {"defense": "krum", "f": f, "m": self.m, "fallback": 1}
        selected_updates = [updates[i] for i in selected]
        return unweighted_mean(selected_updates), {"defense": "krum", "f": f, "m": self.m, "selected": selected}


@register_defense("multi_krum", aliases=["multi-krum", "multikrum"])
class MultiKrumDefense(KrumDefense):
    def __init__(self, f: int | None = None, m: int = 2, n_malicious: int = 0):
        super().__init__(f=f, m=m, n_malicious=n_malicious)
