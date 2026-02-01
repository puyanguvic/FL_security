
from ..base import ByzantineDefense
from ..detection.wbc_core import wbc

class WBCDefense(ByzantineDefense):
    def __init__(self, lr: float, device=None):
        self.lr = lr
        self.device = device

    def defend(self, client_updates, global_weights_before, all_client_indices, **kwargs):
        w_before = kwargs.get("w_before")
        delta_w_before = kwargs.get("delta_w_before")
        w_new = wbc(self.lr, client_updates, w_before, delta_w_before, self.device)
        return w_new, {}
