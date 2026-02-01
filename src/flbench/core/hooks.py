from __future__ import annotations

from typing import Protocol

from flbench.core.context import RunContext


class ClientHook(Protocol):
    def before_round(self, ctx: RunContext, **kwargs) -> None: ...

    def after_round(self, ctx: RunContext, **kwargs) -> None: ...

    def after_backward(self, ctx: RunContext, **kwargs) -> None: ...


class ServerHook(Protocol):
    def before_aggregate(self, ctx: RunContext, **kwargs) -> None: ...

    def after_aggregate(self, ctx: RunContext, **kwargs) -> None: ...
