
from ..base import ByzantineDefense
from ...utils.model_averaging import average_weights, weighted_average_weights

class MeanDefense(ByzantineDefense):
    def defend(self, client_updates, global_weights_before, all_client_indices, **kwargs):
        weights = kwargs.get("weights")
        if weights is not None and len(weights) == len(client_updates):
            return weighted_average_weights(client_updates, weights), {}
        return average_weights(client_updates), {}
