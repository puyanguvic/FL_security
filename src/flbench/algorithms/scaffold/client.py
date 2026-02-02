from __future__ import annotations

import torch

from flbench.algorithms.scaffold.model import ScaffoldModel
from flbench.core.client_base import BaseClient


class ScaffoldClient(BaseClient):
    def __init__(self, args):
        self.local_c = None
        self.global_c = None
        super().__init__(args)

    def build_model(self, task_spec):
        return ScaffoldModel(task_spec.build_model(self.args.model))

    def local_steps(self):
        return max(1, super().local_steps())

    def before_round(self, input_model, global_model):
        self.global_c = self.model.get_c()
        if self.local_c is None:
            self.local_c = {name: torch.zeros_like(param.data) for name, param in self.model.model.named_parameters()}

    def after_backward(self, model, global_model):
        with torch.no_grad():
            for name, param in model.model.named_parameters():
                if param.grad is None:
                    continue
                param.grad.add_(self.global_c[name] - self.local_c[name])

    def after_round(self, global_model, steps):
        global_params = dict(global_model.model.named_parameters())
        local_params = dict(self.model.model.named_parameters())
        delta_c = {}
        scale = 1.0 / (steps * self.args.lr)
        for name, param in local_params.items():
            delta = -self.global_c[name] + scale * (global_params[name].data - param.data)
            delta_c[name] = delta
            self.local_c[name] = self.local_c[name] + delta

        new_global_c = {name: self.global_c[name] + delta_c[name] for name in delta_c}
        self.model.set_c(new_global_c)


def main(args):
    client = ScaffoldClient(args)
    client.run()
