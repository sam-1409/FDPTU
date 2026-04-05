import json
import os

class AuditLogger:
    """
    Append-only audit log for unlearning events
    """

    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                json.dump([], f)

    def log(self, proof):
        with open(self.log_path, "r") as f:
            logs = json.load(f)

        logs.append(proof.to_dict())

        with open(self.log_path, "w") as f:
            json.dump(logs, f, indent=2)