
import copy
from ..base import ByzantineAttack
from ..utils import get_malicious_client_updates
from ...utils.federated_metrics import calculate_gradients, average_gradients

class SMEAttack(ByzantineAttack):
    def __init__(self, learning_rate, surrogate_scale=1.0, attacker_ability="Full"):
        self.lr = learning_rate
        self.scale = surrogate_scale
        self.ability = attacker_ability

    def attack(self, client_updates, global_weights, all_client_indices, malicious_client_indices, **kwargs):
        attacked = copy.deepcopy(client_updates)
        if self.ability == "Part":
            knowledge = get_malicious_client_updates(client_updates, all_client_indices, malicious_client_indices)
        else:
            knowledge = client_updates
        if not knowledge:
            return attacked
        grads = calculate_gradients(global_weights, knowledge, self.lr)
        gavg = average_gradients(grads)
        malicious = {k: global_weights[k] - self.lr * self.scale * gavg[k] for k in gavg}
        for i,cid in enumerate(all_client_indices):
            if cid in malicious_client_indices:
                attacked[i] = malicious
        return attacked
