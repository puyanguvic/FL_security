
import copy
from ..base import ByzantineAttack
from ..utils import get_malicious_client_updates
from ...utils.model_averaging import average_weights

class BackdoorCriticalLayerAttack(ByzantineAttack):
    def __init__(self, critical_layer_names, poison_scale=1.0, attacker_ability="Part"):
        self.layers = critical_layer_names
        self.scale = poison_scale
        self.ability = attacker_ability

    def attack(self, client_updates, global_weights, all_client_indices, malicious_client_indices, **kwargs):
        attacked = copy.deepcopy(client_updates)
        if self.ability == "Part":
            knowledge = get_malicious_client_updates(client_updates, all_client_indices, malicious_client_indices)
        else:
            knowledge = client_updates
        if not knowledge:
            return attacked
        base = average_weights(knowledge)
        for i,cid in enumerate(all_client_indices):
            if cid in malicious_client_indices:
                for k in self.layers:
                    if k in base and k in global_weights:
                        attacked[i][k] = global_weights[k] + self.scale * (base[k] - global_weights[k])
        return attacked
