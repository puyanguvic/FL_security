from __future__ import annotations

from flbench.algorithms.fedavg.client import FedAvgClient
from flbench.algorithms.fedprox.client import FedProxClient
from flbench.algorithms.scaffold.client import ScaffoldClient
from flbench.cli.client_args import parse_client_args

_CLIENTS = {
    "fedavg": FedAvgClient,
    "fedprox": FedProxClient,
    "scaffold": ScaffoldClient,
}


def main(argv: list[str] | None = None) -> None:
    args = parse_client_args(argv)
    algo = str(getattr(args, "algo", "fedavg") or "fedavg").lower()
    if algo not in _CLIENTS:
        raise KeyError(f"Unknown algo '{algo}'. Available: {', '.join(sorted(_CLIENTS))}")
    client = _CLIENTS[algo](args)
    client.run()


if __name__ == "__main__":
    main()
