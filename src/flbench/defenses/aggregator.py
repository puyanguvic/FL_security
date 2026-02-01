from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.app_constant import AppConstants
try:
    from nvflare.apis.shareable import Shareable
except ImportError:  # pragma: no cover - fallback for older nvflare
    from nvflare.app_common.app_defined.shareable import Shareable
try:
    from nvflare.apis.fl_constant import ReservedKey
except ImportError:  # pragma: no cover - fallback for older nvflare
    from nvflare.apis.shareable import ReservedHeaderKey as ReservedKey

from flbench.defenses.base import Defense, DefenseContext, weighted_mean


class DefenseAggregator(Aggregator):
    def __init__(
        self,
        defense: Defense | None = None,
        expected_data_kind: DataKind = DataKind.WEIGHT_DIFF,
        aggregation_weights: Dict[str, float] | None = None,
        weigh_by_local_iter: bool = True,
    ):
        super().__init__()
        self.defense = defense
        self.expected_data_kind = expected_data_kind
        self.aggregation_weights = aggregation_weights or {}
        self.weigh_by_local_iter = weigh_by_local_iter
        self._reset()

    def _reset(self):
        self._updates: List[Dict[str, torch.Tensor]] = []
        self._weights: List[float] = []
        self._contributors: List[str] = []
        self._processed_algorithm = None

    def reset(self, fl_ctx: FLContext) -> None:
        self._reset()

    def handle_event(self, event_type: str, fl_ctx: FLContext) -> None:
        if event_type == EventType.START_RUN:
            self._reset()

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        dxo = from_shareable(shareable)
        if dxo.data_kind != self.expected_data_kind:
            self.log_error(fl_ctx, f"Expected {self.expected_data_kind} but got {dxo.data_kind}")
            return False

        rc = shareable.get_return_code()
        if rc and rc != ReturnCode.OK:
            self.log_error(fl_ctx, f"Shareable has return code {rc}")
            return False

        contributor = shareable.get_peer_prop(ReservedKey.IDENTITY_NAME, "unknown")
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        contribution_round = shareable.get_cookie(AppConstants.CONTRIBUTION_ROUND)
        if contribution_round is not None and current_round != contribution_round:
            self.log_warning(
                fl_ctx,
                f"Discarding contribution from {contributor} for round {contribution_round} (current {current_round})",
            )
            return False

        if contributor in self._contributors:
            self.log_warning(fl_ctx, f"Duplicate contribution from {contributor}. Discarding.")
            return False

        data = dxo.data
        if not isinstance(data, dict):
            self.log_error(fl_ctx, f"Unexpected data type {type(data)} for contributor {contributor}")
            return False

        weight = self.aggregation_weights.get(contributor, 1.0)
        if self.weigh_by_local_iter:
            n_iter = dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND)
            if n_iter is not None:
                weight = weight * float(n_iter)

        update = {k: self._to_tensor(v) for k, v in data.items()}
        self._updates.append(update)
        self._weights.append(float(weight))
        self._contributors.append(contributor)

        processed_alg = dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM)
        if processed_alg:
            self._processed_algorithm = processed_alg

        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        if not self._updates:
            self.log_warning(fl_ctx, "No contributions received for aggregation.")
            dxo = DXO(data_kind=self.expected_data_kind, data={})
            return dxo.to_shareable()

        ctx = DefenseContext(
            weights=self._weights,
            contributors=self._contributors,
            current_round=fl_ctx.get_prop(AppConstants.CURRENT_ROUND),
            n_clients=fl_ctx.get_prop(getattr(AppConstants, "NUM_TOTAL_CLIENTS", None))
            if getattr(AppConstants, "NUM_TOTAL_CLIENTS", None)
            else None,
        )

        if self.defense is None:
            aggregated = weighted_mean(self._updates, self._weights)
            info = {"defense": "none"}
        else:
            aggregated, info = self.defense.apply(self._updates, ctx)

        dxo = DXO(data_kind=self.expected_data_kind, data=aggregated)
        if self._processed_algorithm:
            dxo.set_meta_prop(MetaKey.PROCESSED_ALGORITHM, self._processed_algorithm)
        if info:
            dxo.set_meta_prop("defense_info", info)

        self._reset()
        return dxo.to_shareable()

    def _to_tensor(self, value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        return torch.tensor(value)
