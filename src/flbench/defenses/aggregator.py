from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np

from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.abstract.model import ModelLearnableKey
from nvflare.apis.dxo import DataKind, DXO, from_shareable
from nvflare.apis.fl_constant import FLMetaKey, ReservedKey

from flbench.utils.model_averaging import average_weights, weighted_average_weights


class DefenseAggregator(InTimeAccumulateWeightedAggregator):
    """Wrapper aggregator that applies defense over collected client updates."""

    def __init__(
        self,
        defense: Any,
        expected_data_kind: Any,
        aggregation_weights: Dict[str, float],
        weigh_by_local_iter: bool = True,
        attack: Any | None = None,
        attack_name: str | None = None,
        malicious_site_indices: Set[str] | list[str] | None = None,
    ) -> None:
        super().__init__(
            expected_data_kind=expected_data_kind,
            aggregation_weights=aggregation_weights,
            weigh_by_local_iter=weigh_by_local_iter,
        )
        self.defense = defense
        self.attack = attack
        self.attack_name = attack_name
        if malicious_site_indices is None:
            self.malicious_site_indices: List[str] = []
        else:
            self.malicious_site_indices = [str(v) for v in malicious_site_indices]
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

    def _to_numpy_state_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in data.items():
            if hasattr(v, "detach"):
                out[k] = v.detach().cpu().numpy()
            else:
                out[k] = np.asarray(v)
        return out

    def _canonical_site_name(self, name: Any) -> str:
        s = str(name)
        match = re.search(r"(site-\\d+)", s)
        if match:
            return match.group(1)
        match = re.search(r"(\\d+)$", s)
        if match:
            return f"site-{match.group(1)}"
        return s

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
        n_iter = dxo.get_meta_prop(FLMetaKey.NUM_STEPS_CURRENT_ROUND)
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
        if (self.defense is None and self.attack is None) or not self._defense_updates:
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

        if self.attack is not None and self.malicious_site_indices:
            malicious_set = {self._canonical_site_name(v) for v in self.malicious_site_indices}
            all_names = [self._canonical_site_name(u.get("contributor_name", "?")) for u in self._defense_updates]
            client_weights = self.attack.attack(
                client_updates=client_weights,
                global_weights=global_weights,
                all_client_indices=all_names,
                malicious_client_indices=malicious_set,
            )
            if self.attack_name:
                self.logger.info(f"Applied byzantine attack: {self.attack_name}")

        if self.defense is not None:
            all_indices = list(range(len(client_weights)))
            defended_weights, info = self.defense.defend(
                client_updates=client_weights,
                global_weights_before=global_weights,
                all_client_indices=all_indices,
                weights=weights,
            )
            if info:
                self.logger.info(f"Defense info: {info}")
        else:
            if weights is not None and len(weights) == len(client_weights):
                defended_weights = weighted_average_weights(client_weights, weights)
            else:
                defended_weights = average_weights(client_weights)

        if data_kind == DataKind.WEIGHT_DIFF:
            defended_data = self._weights_to_diff(defended_weights, global_weights)
        else:
            defended_data = defended_weights

        defended_data = self._to_numpy_state_dict(defended_data)

        self._defense_updates = []
        for agg in self.dxo_aggregators.values():
            agg.reset_aggregation_helper()
        dxo = DXO(data_kind=data_kind, data=defended_data)
        return dxo.to_shareable()
