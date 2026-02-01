
import copy
from scipy.stats import norm
from attacks.base import ByzantineAttack
from utils.federated_metrics import average_gradients, calculate_gradient_std_dev

class LIEAttack(ByzantineAttack):
    def attack(self, client_updates, global_weights, all_client_indices, malicious_client_indices, **kwargs):
        attacked = copy.deepcopy(client_updates)
        benign = [u for i,u in enumerate(client_updates) if all_client_indices[i] not in malicious_client_indices]
        if not benign:
            return attacked
        avg = average_gradients(benign)
        std = calculate_gradient_std_dev(benign)
        n, f = len(all_client_indices), len(malicious_client_indices)
        s = (n // 2 + 1) - f
        z = norm.ppf((n - f - s) / (n - f))
        malicious = {k: avg[k] + z * std[k] for k in avg}
        for i, cid in enumerate(all_client_indices):
            if cid in malicious_client_indices:
                attacked[i] = malicious
        return attacked
