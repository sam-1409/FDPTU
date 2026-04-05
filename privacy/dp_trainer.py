import torch
import torch.nn.functional as F
from opacus import PrivacyEngine


class DPClientTrainer:
    def __init__(
        self,
        model,
        optimizer,
        dataloader,
        max_grad_norm=1.0,
        noise_multiplier=1.0,
        device="cpu"
    ):
        self.device = device
        self.model = model.to(device)
        self.model.train()   # 🔥 CRITICAL FIX
        self.optimizer = optimizer

        self.privacy_engine = PrivacyEngine()

        self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )


        self.current_epsilon = None

    def train_one_epoch(self, dataloader=None):
        self.model.train()
        total_loss = 0.0

        for batch in self.dataloader:
            self.optimizer.zero_grad()

            if "iris" in batch:
                iris = batch["iris"].to(self.device)
                fingerprint = batch["fingerprint"].to(self.device)

                logits = self.model(iris, fingerprint)

                labels = torch.zeros(
                    logits.size(0),
                    dtype=torch.long,
                    device=self.device
                )

                loss = F.cross_entropy(logits, labels)

            else:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(images)

                class_weights = torch.tensor(
                    [1.0, 3.0],
                    device=self.device
                )

                loss = F.cross_entropy(
                    logits,
                    labels,
                    weight=class_weights
                )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        self.current_epsilon = self.privacy_engine.get_epsilon(delta=1e-5)

        return total_loss / len(self.dataloader)

    def get_epsilon(self, delta=1e-5):
        return self.privacy_engine.get_epsilon(delta)