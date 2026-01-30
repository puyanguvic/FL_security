from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ScaffoldModel(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model
        self._c_key_map: Dict[str, str] = {}
        for name, param in base_model.named_parameters():
            buf_name = _c_buf_name(name)
            self.register_buffer(buf_name, torch.zeros_like(param.data))
            self._c_key_map[name] = buf_name

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_c(self) -> Dict[str, torch.Tensor]:
        return {name: getattr(self, buf_name) for name, buf_name in self._c_key_map.items()}

    def set_c(self, c: Dict[str, torch.Tensor]) -> None:
        for name, buf_name in self._c_key_map.items():
            if name in c:
                buf = getattr(self, buf_name)
                buf.data.copy_(c[name])


def _c_buf_name(param_name: str) -> str:
    return f"scaffold_c__{param_name.replace('.', '__')}"
