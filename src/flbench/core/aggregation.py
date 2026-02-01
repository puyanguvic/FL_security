from __future__ import annotations

from typing import Any, Dict

from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator

from flbench.defenses import build_defense_from_args
from flbench.defenses.aggregator import DefenseAggregator


def build_nvflare_aggregator(*, args: Any, n_clients: int, expected_data_kind: DataKind) -> Any:
    """Build an NVFlare aggregator with optional defense wrapping.

    Design goal:
      - algorithms/*/server.py should NOT have to know defense details.
      - adding a new defense or robust aggregator should NOT require touching algorithm code.

    Notes:
      - We always set per-site aggregation_weights to avoid NVFlare warnings.
      - Actual FedAvg weighting is driven by MetaKey.NUM_STEPS_CURRENT_ROUND sent by clients.
    """

    defense = build_defense_from_args(args)
    aggregation_weights: Dict[str, float] = {f"site-{i}": 1.0 for i in range(1, n_clients + 1)}

    if defense is None:
        return InTimeAccumulateWeightedAggregator(
            expected_data_kind=expected_data_kind,
            aggregation_weights=aggregation_weights,
        )

    return DefenseAggregator(
        defense=defense,
        expected_data_kind=expected_data_kind,
        aggregation_weights=aggregation_weights,
        weigh_by_local_iter=True,
    )
