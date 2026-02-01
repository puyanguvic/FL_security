
from typing import List, Set
from .base import StateDict

def get_malicious_client_updates(all_updates: List[StateDict], all_client_indices: List[int], malicious_client_indices: Set[int]):
    return [all_updates[i] for i, cid in enumerate(all_client_indices) if cid in malicious_client_indices]
