import torch
import torch.nn.functional as F

class ClientTrainer:
    def __init__(self, model, optimizer, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            self.optimizer.zero_grad()

            if "iris" in batch:
                iris = batch["iris"].to(self.device)
                fp = batch["fingerprint"].to(self.device)
                logits = self.model(iris, fp)
                labels = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)
            else:
                images = batch["image"].to(self.device)
                logits = self.model(images)
                labels = batch["label"].to(self.device)

            class_weights = torch.tensor([1.0, 3.0], device=labels.device)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)