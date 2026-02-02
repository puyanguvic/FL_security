from __future__ import annotations

import torch

from flbench.core.client_base import BaseClient


class FedProxClient(BaseClient):
    def _prox_term(self, model, global_model) -> torch.Tensor:
        prox = torch.zeros(1, device=self.device)
        for p, g in zip(model.parameters(), global_model.parameters(), strict=False):
            prox = prox + (p - g).pow(2).sum()
        return prox

    def compute_loss(self, outputs, labels, model, global_model):
        loss = self.criterion(outputs, labels)
        if self.args.prox_mu > 0:
            loss = loss + (self.args.prox_mu / 2.0) * self._prox_term(model, global_model)
        return loss


def main(args):
    client = FedProxClient(args)
    client.run()
