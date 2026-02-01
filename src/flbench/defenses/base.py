
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import torch

StateDict = Dict[str, torch.Tensor]

class ByzantineDefense(ABC):
    @abstractmethod
    def defend(
        self,
        client_updates: List[StateDict],
        global_weights_before: StateDict,
        all_client_indices: List[int],
        **kwargs,
    ) -> Tuple[StateDict, Dict[str, Any]]:
        pass
