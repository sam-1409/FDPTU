import json
import os
import time

class MetricLogger:
    def __init__(self, save_path):
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if not os.path.exists(save_path):
            with open(save_path, "w") as f:
                json.dump([], f)

    def log(
        self,
        round_id,
        method,
        metrics,
        epsilon=None,
        mia_auc=None,
        train_loss=None,
        val_loss=None
    ):
        entry = {
            "round": round_id,
            "method": method,
            **metrics,
            "epsilon": epsilon,
            "mia_auc": mia_auc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "timestamp": time.time()
        }

        with open(self.save_path, "r") as f:
            logs = json.load(f)

        logs.append(entry)

        with open(self.save_path, "w") as f:
            json.dump(logs, f, indent=2)