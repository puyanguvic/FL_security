from __future__ import annotations

from flbench.core.client_base import BaseClient


class FedAvgClient(BaseClient):
    pass


def main(args):
    client = FedAvgClient(args)
    client.run()
