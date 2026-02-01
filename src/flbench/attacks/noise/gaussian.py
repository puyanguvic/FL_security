
import copy, torch
from attacks.base import ByzantineAttack

class GaussianAttack(ByzantineAttack):
    def __init__(self, attack_scale: float, device: torch.device):
        self.attack_scale = attack_scale
        self.device = device

    def attack(self, client_updates, global_weights, all_client_indices, malicious_client_indices, **kwargs):
        attacked = copy.deepcopy(client_updates)
        for i, cid in enumerate(all_client_indices):
            if cid in malicious_client_indices:
                for k, v in attacked[i].items():
                    if v.dtype.is_floating_point:
                        attacked[i][k] = v + torch.randn_like(v) * self.attack_scale
        return attacked
