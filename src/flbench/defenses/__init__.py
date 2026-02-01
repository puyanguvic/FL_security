
from defenses.aggregation.mean import MeanDefense
from defenses.aggregation.multikrum import MultiKrumDefense
from defenses.aggregation.wbc import WBCDefense
from defenses.detection.fgnv import FGNVDefense
from defenses.detection.fldetector import FLDetectorDefense
from defenses.reputation.beta import BetaReputationDefense

DEFENSE_REGISTRY = {
    "mean": MeanDefense,
    "multikrum": MultiKrumDefense,
    "wbc": WBCDefense,
    "fgnv": FGNVDefense,
    "fldetector": FLDetectorDefense,
    "beta_reputation": BetaReputationDefense,
}
