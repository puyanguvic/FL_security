
from defenses.base import ByzantineDefense

class FLDetectorDefense(ByzantineDefense):
    def __init__(self, detector):
        self.detector = detector

    def defend(self, client_updates, global_weights_before, all_client_indices, **kwargs):
        malicious, scores = self.detector.step_and_detect(**kwargs)
        return kwargs.get("global_weights_after"), {"malicious": malicious, "scores": scores}
