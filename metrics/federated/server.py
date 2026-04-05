import copy
import random
import torch

from FDPU.federated.aggregation import fedavg
from FDPU.metrics.evaluation import evaluate_global_model

class FederatedServer:
    def __init__(
        self,
        global_model,
        client_loaders,
        client_trainer_fn,
        rounds=10,
        client_fraction=0.3,
        device="cpu",
        metric_logger=None,
        val_loader=None,
        method_name="FedAvg"
    ):
        self.global_model = global_model
        self.client_loaders = client_loaders
        self.client_trainer_fn = client_trainer_fn
        self.rounds = rounds
        self.client_fraction = client_fraction
        self.device = device

        self.metric_logger = metric_logger
        self.val_loader = val_loader
        self.method_name = method_name

        self.checkpoints = {}
        self.round_logs = {}
        self.initial_state = copy.deepcopy(global_model.state_dict())

    def train(self):
        client_ids = list(self.client_loaders.keys())

        for rnd in range(1, self.rounds + 1):
            print(f"\n--- Federated Round {rnd} ---")

            selected = random.sample(
                client_ids,
                max(1, int(len(client_ids) * self.client_fraction))
            )

            self.round_logs[rnd] = selected
            local_states = []

            for cid in selected:
                local_model = copy.deepcopy(self.global_model)
                trainer = self.client_trainer_fn(
                    local_model,
                    self.client_loaders[cid]
                )
                trainer.train_one_epoch(self.client_loaders[cid])
                local_states.append(local_model.state_dict())

            new_global_state = fedavg(local_states)
            self.global_model.load_state_dict(new_global_state)
            self.checkpoints[rnd] = copy.deepcopy(new_global_state)

            # ===== METRIC LOGGING (STEP 10.1) =====
            if self.metric_logger and self.val_loader:
                metrics = evaluate_global_model(
                    self.global_model,
                    self.val_loader,
                    self.device
                )
                self.metric_logger.log(
                    round_id=rnd,
                    method=self.method_name,
                    metrics=metrics
                )