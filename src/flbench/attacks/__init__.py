
from .registry import AttackContext, build_attack_from_args, diff_l2_norm, list_attacks

# Optional: byzantine attack registry (used outside the client-side update attacks)
try:
    from .noise.gaussian import GaussianAttack
    from .statistical.lie import LIEAttack
    from .model_poisoning.fang import FangAttack
    from .model_poisoning.sme import SMEAttack
    from .model_poisoning.backdoor_layers import BackdoorCriticalLayerAttack
    from .optimization.minmax import MinMaxAttack
    from .optimization.minsum import MinSumAttack

    ATTACK_REGISTRY = {
        "gaussian": GaussianAttack,
        "lie": LIEAttack,
        "fang": FangAttack,
        "sme": SMEAttack,
        "backdoor": BackdoorCriticalLayerAttack,
        "minmax": MinMaxAttack,
        "minsum": MinSumAttack,
    }
except Exception:
    ATTACK_REGISTRY = {}

__all__ = [
    "AttackContext",
    "build_attack_from_args",
    "diff_l2_norm",
    "list_attacks",
    "ATTACK_REGISTRY",
]
