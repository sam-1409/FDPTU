import torch
import torch.nn.functional as F

class AttackMonitor:
    """
    Monitors client updates and detects malicious behavior
    """

    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.attack_log = []

    def evaluate(self, logits, round_id, client_id):
        probs = F.softmax(logits, dim=1)
        attack_confidence = probs[:, 1].mean().item()

        if attack_confidence >= self.threshold:
            self.attack_log.append({
                "round": round_id,
                "client_id": client_id,
                "confidence": attack_confidence
            })
            return True, attack_confidence

        return False, attack_confidence