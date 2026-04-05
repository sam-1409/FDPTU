import sys
sys.path.insert(0, "/content/drive/MyDrive")

import torch
from torch.utils.data import DataLoader

from FDPTU.models.face_client_model import FaceClientModel
from FDPTU.datasets.face_attack_dataset import FaceAttackDataset
from FDPTU.datasets.transforms import face_transform
from FDPTU.attacks.attack_monitor import AttackMonitor
from FDPTU.utils.seed import set_seed

def main():
    set_seed(42)

    DATA_ROOT = "/content/drive/MyDrive/datasets/face_attack"
    client_id = "FACE_001"

    dataset = FaceAttackDataset(
        DATA_ROOT,
        client_id,
        transform=face_transform()
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = FaceClientModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    monitor = AttackMonitor(threshold=0.8)

    model.train()
    for epoch in range(3):
        for batch in loader:
            optimizer.zero_grad()
            images = batch["image"]
            labels = batch["label"]

            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            detected, conf = monitor.evaluate(
                logits, round_id=epoch, client_id=client_id
            )

        print(f"Epoch {epoch}: Attack detected={detected}, confidence={conf:.3f}")

if __name__ == "__main__":
    main()