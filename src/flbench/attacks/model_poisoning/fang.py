
import copy, torch
from ..base import ByzantineAttack

class FangAttack(ByzantineAttack):
    def attack(self, client_updates, global_weights, all_client_indices, malicious_client_indices, **kwargs):
        attacked = copy.deepcopy(client_updates)
        benign = [u for i,u in enumerate(client_updates) if all_client_indices[i] not in malicious_client_indices]
        if not benign:
            return attacked
        bn_keys = ("running_mean","running_var","num_batches_tracked")
        mins, maxs, avg = {}, {}, {}
        for k,v in benign[0].items():
            mins[k], maxs[k] = v.clone(), v.clone()
            avg[k] = v.clone().float() if not any(b in k for b in bn_keys) else v.clone()
        for u in benign[1:]:
            for k,v in u.items():
                mins[k] = torch.minimum(mins[k], v)
                maxs[k] = torch.maximum(maxs[k], v)
                if not any(b in k for b in bn_keys):
                    avg[k] += v.float()
        for k in avg:
            if not any(b in k for b in bn_keys):
                avg[k] /= len(benign)
        direction = {k: torch.sign(avg[k] - global_weights[k]) for k in avg}
        malicious = {k: torch.where(direction[k] > 0, mins[k], maxs[k]) for k in mins}
        for i,cid in enumerate(all_client_indices):
            if cid in malicious_client_indices:
                attacked[i] = malicious
        return attacked
