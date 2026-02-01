from __future__ import annotations

from typing import Any, Dict

from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator


class DefenseAggregator(InTimeAccumulateWeightedAggregator):
    """Wrapper aggregator that currently falls back to standard aggregation.

    NOTE: This is a placeholder implementation so that simulations without defenses run.
    If you intend to use server-side defenses, extend this to apply the defense
    over collected client updates before returning the aggregated DXO.
    """

    def __init__(
        self,
        defense: Any,
        expected_data_kind: Any,
        aggregation_weights: Dict[str, float],
        weigh_by_local_iter: bool = True,
    ) -> None:
        super().__init__(
            expected_data_kind=expected_data_kind,
            aggregation_weights=aggregation_weights,
            weigh_by_local_iter=weigh_by_local_iter,
        )
        self.defense = defense
        if defense is not None:
            self.logger.warning(
                "DefenseAggregator is currently a pass-through. "
                "Defenses are not applied in this implementation."
            )
