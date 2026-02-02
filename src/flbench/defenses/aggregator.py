from __future__ import annotations

from typing import Any, Dict, List

from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.abstract.model import ModelLearnableKey
from nvflare.apis.dxo import DataKind, DXO, from_shareable
from nvflare.apis.fl_constant import MetaKey, ReservedKey


class DefenseAggregator(InTimeAccumulateWeightedAggregator):
    """Wrapper aggregator that applies defense over collected client updates."""

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
        self._defense_updates: List[dict] = []

    def _to_torch_state_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        out: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu()
            else:
                out[k] = torch.as_tensor(v)
        return out

    def _get_global_weights(self, fl_ctx) -> Dict[str, Any]:
        global_model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        if global_model is None:
            return {}
        weights = global_model.get(ModelLearnableKey.WEIGHTS, {})
        if not isinstance(weights, dict):
            return {}
        return self._to_torch_state_dict(weights)

    def _diff_to_weights(self, diff: Dict[str, Any], global_weights: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, gv in global_weights.items():
            if k in diff:
                out[k] = gv + diff[k]
            else:
                out[k] = gv.clone() if hasattr(gv, "clone") else gv
        return out

    def _weights_to_diff(self, weights: Dict[str, Any], global_weights: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, gv in global_weights.items():
            if k in weights:
                out[k] = weights[k] - gv
            else:
                out[k] = gv.clone().zero_() if hasattr(gv, "clone") else gv
        return out

    def accept(self, shareable, fl_ctx) -> bool:
        accepted = super().accept(shareable, fl_ctx)
        if not accepted:
            return False

        try:
            dxo = from_shareable(shareable)
        except Exception:
            return True

        if dxo.data_kind == DataKind.COLLECTION:
            # Only support single DXO for now.
            return True

        contributor_name = shareable.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default="?")
        n_iter = dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND)
        if n_iter is None:
            n_iter = 1.0

        aggregation_weights = self.aggregation_weights.get(self._single_dxo_key, self.aggregation_weights)
        aggregation_weight = aggregation_weights.get(contributor_name, 1.0)

        self._defense_updates.append(
            {
                "contributor_name": contributor_name,
                "data_kind": dxo.data_kind,
                "data": self._to_torch_state_dict(dxo.data),
                "weight": float(aggregation_weight) * float(n_iter),
            }
        )
        return True

    def aggregate(self, fl_ctx):
        if self.defense is None or not self._defense_updates:
            return super().aggregate(fl_ctx)

        # Only handle single DXO for now.
        data_kind = self._defense_updates[0]["data_kind"]
        global_weights = self._get_global_weights(fl_ctx)
        if not global_weights:
            return super().aggregate(fl_ctx)

        client_updates = [u["data"] for u in self._defense_updates]
        weights = [u["weight"] for u in self._defense_updates]

        if data_kind == DataKind.WEIGHT_DIFF:
            client_weights = [self._diff_to_weights(u, global_weights) for u in client_updates]
        elif data_kind == DataKind.WEIGHTS:
            client_weights = client_updates
        else:
            return super().aggregate(fl_ctx)

        all_indices = list(range(len(client_weights)))
        defended_weights, info = self.defense.defend(
            client_updates=client_weights,
            global_weights_before=global_weights,
            all_client_indices=all_indices,
            weights=weights,
        )
        if info:
            self.logger.info(f"Defense info: {info}")

        if data_kind == DataKind.WEIGHT_DIFF:
            defended_data = self._weights_to_diff(defended_weights, global_weights)
        else:
            defended_data = defended_weights

        self._defense_updates = []
        dxo = DXO(data_kind=data_kind, data=defended_data)
        return dxo.to_shareable()
