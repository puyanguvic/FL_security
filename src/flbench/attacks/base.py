
from abc import ABC, abstractmethod
from typing import Dict, List, Set
import torch

StateDict = Dict[str, torch.Tensor]

class ByzantineAttack(ABC):
    @abstractmethod
    def attack(self, client_updates: List[StateDict], global_weights: StateDict,
               all_client_indices: List[int], malicious_client_indices: Set[int], **kwargs) -> List[StateDict]:
        pass
