
from ..base import ByzantineAttack
from ...utils.federated_metrics import calculate_gradients, average_gradients, calculate_l2_norm, distance

class MinSumAttack(ByzantineAttack):
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def attack(self, client_updates, global_weights, all_client_indices, malicious_client_indices, **kwargs):
        attacked = [u.copy() for u in client_updates]
        grads = calculate_gradients(global_weights, client_updates, self.lr)
        dist_sum = {i: sum(distance(grads, grads[i]).values()) for i in range(len(grads))}
        gavg = average_gradients(grads)
        norm = calculate_l2_norm(gavg)
        direction = {k: -gavg[k]/norm for k in gavg}
        gamma, last, step = 0.5, 2.5, 100.0
        while abs(gamma-last) > 1e-3:
            nabla = {k: direction[k]*gamma + gavg[k] for k in gavg}
            delta = {i: calculate_l2_norm(grads[i], nabla) for i in grads}
            last = gamma
            gamma += step/2 if sum(v*v for v in delta.values()) < max(dist_sum.values())**2 else -step/2
            step /= 2
        for i,cid in enumerate(all_client_indices):
            if cid in malicious_client_indices:
                for k in nabla:
                    attacked[i][k] = global_weights[k] - self.lr * nabla[k]
        return attacked
