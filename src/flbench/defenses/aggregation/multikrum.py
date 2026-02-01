
from defenses.base import ByzantineDefense
from utils.model_averaging import average_weights
from utils.federated_metrics import calculate_gradients, calculate_l2_norm
import numpy as np

class MultiKrumDefense(ByzantineDefense):
    def __init__(self, num_attackers: int, num_benign_to_select: int, learning_rate: float):
        self.f = num_attackers
        self.m = num_benign_to_select
        self.lr = learning_rate

    def defend(self, client_updates, global_weights_before, all_client_indices, **kwargs):
        n = len(client_updates)
        if n <= self.f:
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
        return average_weights(selected), {"selected_indices": idx.tolist()}
