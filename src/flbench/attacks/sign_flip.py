from __future__ import annotations

from flbench.attacks.base import Attack, AttackContext, apply_to_diff
from flbench.attacks.registry import register_attack


@register_attack("sign_flip", aliases=["sign-flip", "signflip"])
class SignFlipAttack(Attack):
    def apply(self, diff, ctx: AttackContext):
        return apply_to_diff(diff, lambda t: -t), {"attack": "sign_flip"}
