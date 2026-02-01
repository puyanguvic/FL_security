
from defenses.base import ByzantineDefense
from utils.model_averaging import average_weights

class MeanDefense(ByzantineDefense):
    def defend(self, client_updates, global_weights_before, all_client_indices, **kwargs):
        return average_weights(client_updates), {}
