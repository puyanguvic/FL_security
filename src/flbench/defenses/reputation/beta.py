
from defenses.base import ByzantineDefense
from typing import List
import numpy as np

class BetaReputationDefense(ByzantineDefense):
    def __init__(self, discount: float):
        self.discount = discount

    def defend(self, client_updates, global_weights_before, all_client_indices, **kwargs):
        malicious = kwargs.get("malicious", [])
        alpha = kwargs["alpha"]
        beta = kwargs["beta"]

        alpha_new, beta_new = alpha.copy(), beta.copy()
        rep = []

        for uid in all_client_indices:
            if uid in malicious:
                alpha_new[uid] = self.discount * alpha[uid]
                beta_new[uid] = self.discount * beta[uid] + 1
                rep.append(0.0)
            else:
                alpha_new[uid] = self.discount * alpha[uid] + 1
                beta_new[uid] = self.discount * beta[uid]
                rep.append(alpha_new[uid] / (alpha_new[uid] + beta_new[uid]))

        return global_weights_before, {
            "reputation": np.array(rep),
            "alpha": alpha_new,
            "beta": beta_new,
        }
