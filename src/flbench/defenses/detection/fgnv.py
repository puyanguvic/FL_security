
from ..base import ByzantineDefense
from ...utils.federated_metrics import calculate_gradients
import torch, numpy as np

class FGNVDefense(ByzantineDefense):
    def __init__(self, learning_rate: float):
        self.lr = learning_rate

    def defend(self, client_updates, global_weights_before, all_client_indices, **kwargs):
        grads = calculate_gradients(global_weights_before, client_updates, self.lr)
        C = len(grads)
        keys = list(grads[0].keys())
        G = torch.tensor([[grads[i][k].norm().item() if torch.is_floating_point(grads[i][k]) else 0.0
                           for k in keys] for i in range(C)])
        inv_mean = (1.0 / (G + 1e-12)).mean(dim=0)
        B = G * inv_mean
        med = B.median(dim=0).values
        suspicious = ((B > med * 1.04) | (B < med * 0.96)).sum(dim=1)
        maxc = suspicious.max().item()
        mal = [all_client_indices[i] for i,c in enumerate(suspicious) if c == maxc and c > 0]
        return global_weights_before, {"malicious": mal}
