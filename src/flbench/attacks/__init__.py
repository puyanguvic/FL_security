
from attacks.noise.gaussian import GaussianAttack
from attacks.statistical.lie import LIEAttack
from attacks.model_poisoning.fang import FangAttack
from attacks.model_poisoning.sme import SMEAttack
from attacks.model_poisoning.backdoor_layers import BackdoorCriticalLayerAttack
from attacks.optimization.minmax import MinMaxAttack
from attacks.optimization.minsum import MinSumAttack

ATTACK_REGISTRY = {
    "gaussian": GaussianAttack,
    "lie": LIEAttack,
    "fang": FangAttack,
    "sme": SMEAttack,
    "backdoor": BackdoorCriticalLayerAttack,
    "minmax": MinMaxAttack,
    "minsum": MinSumAttack,
}
