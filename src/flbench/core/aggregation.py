from __future__ import annotations

from typing import Any, Dict, List, Set

import random

from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator

from flbench.attacks import build_byzantine_attack_from_args, is_byzantine_attack_name
from flbench.defenses import build_defense_from_args
from flbench.defenses.aggregator import DefenseAggregator


def _compute_malicious_set(
    *,
    num_clients: int,
    n_malicious: int,
    mode: str,
    seed: int | None,
    fallback_seed: int | None,
) -> Set[int]:
    if n_malicious <= 0:
        return set()
    if num_clients <= 0:
        return set(range(1, n_malicious + 1))
    if n_malicious > num_clients:
        return set(range(1, n_malicious + 1))
    if mode == "random":
        rng_seed = seed if seed is not None else (fallback_seed if fallback_seed is not None else 0)
        rng = random.Random(int(rng_seed))
        return set(rng.sample(range(1, num_clients + 1), k=n_malicious))
    return set(range(1, n_malicious + 1))


def _compute_malicious_sites(
    *,
    num_clients: int,
    n_malicious: int,
    mode: str,
    seed: int | None,
    fallback_seed: int | None,
) -> List[str]:
    indices = _compute_malicious_set(
        num_clients=num_clients,
        n_malicious=n_malicious,
        mode=mode,
        seed=seed,
        fallback_seed=fallback_seed,
    )
    if not indices:
        return []
    return [f"site-{i}" for i in sorted(indices)]


def build_nvflare_aggregator(*, args: Any, num_clients: int, expected_data_kind: DataKind) -> Any:
    """Build an NVFlare aggregator with optional defense wrapping.

    Design goal:
      - algorithms/*/server.py should NOT have to know defense details.
      - adding a new defense or robust aggregator should NOT require touching algorithm code.

    Notes:
      - We always set per-site aggregation_weights to avoid NVFlare warnings.
      - Actual FedAvg weighting is driven by FLMetaKey.NUM_STEPS_CURRENT_ROUND sent by clients.
    """

    defense = build_defense_from_args(args)
    attack = build_byzantine_attack_from_args(args)
    attack_name = None
    malicious_sites: List[str] = []
    if attack is not None and is_byzantine_attack_name(getattr(args, "attack", None)):
        attack_name = str(getattr(args, "attack", "none"))
        malicious_sites = _compute_malicious_sites(
            num_clients=num_clients,
            n_malicious=int(getattr(args, "n_malicious", 0) or 0),
            mode=str(getattr(args, "malicious_mode", "first")).lower(),
            seed=getattr(args, "malicious_seed", None),
            fallback_seed=getattr(args, "seed", None),
        )
    aggregation_weights: Dict[str, float] = {f"site-{i}": 1.0 for i in range(1, num_clients + 1)}

    if defense is None and attack is None:
        return InTimeAccumulateWeightedAggregator(
            expected_data_kind=expected_data_kind,
            aggregation_weights=aggregation_weights,
        )

    return DefenseAggregator(
        defense=defense,
        expected_data_kind=expected_data_kind,
        aggregation_weights=aggregation_weights,
        weigh_by_local_iter=True,
        attack=attack,
        attack_name=attack_name,
        malicious_site_indices=malicious_sites,
    )
