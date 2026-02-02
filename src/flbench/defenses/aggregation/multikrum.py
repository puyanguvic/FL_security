
from ..base import ByzantineDefense
from ...utils.model_averaging import average_weights, weighted_average_weights
from ...utils.federated_metrics import calculate_gradients, calculate_l2_norm
import numpy as np

class MultiKrumDefense(ByzantineDefense):
    def __init__(self, num_attackers: int, num_benign_to_select: int, learning_rate: float):
        # Keep constructor arg names for NVFLARE component serialization.
        self.num_attackers = num_attackers
        self.num_benign_to_select = num_benign_to_select
        self.learning_rate = learning_rate
        # Internal aliases for brevity.
        self.f = num_attackers
        self.m = num_benign_to_select
        self.lr = learning_rate

    def defend(self, client_updates, global_weights_before, all_client_indices, **kwargs):
        n = len(client_updates)
        if n <= self.f:
            weights = kwargs.get("weights")
            if weights is not None and len(weights) == len(client_updates):
                return weighted_average_weights(client_updates, weights), {}
            return average_weights(client_updates), {}

        grads = calculate_gradients(global_weights_before, client_updates, self.lr)
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = calculate_l2_norm(grads[i], grads[j]) ** 2
                dists[i,j] = dists[j,i] = d

        k = n - self.f - 2
        scores = [np.sum(np.sort(dists[i])[1:k+1]) for i in range(n)]
        idx = np.argsort(scores)[: self.m]
        selected = [client_updates[i] for i in idx]
        weights = kwargs.get("weights")
        if weights is not None and len(weights) == len(client_updates):
            selected_weights = [weights[i] for i in idx]
            return weighted_average_weights(selected, selected_weights), {"selected_indices": idx.tolist()}
        return average_weights(selected), {"selected_indices": idx.tolist()}
