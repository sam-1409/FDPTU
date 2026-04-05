import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from FDPTU.datasets.face_attack_dataset import FaceAttackDataset
from FDPTU.models.face_client_model import FaceClientModel
from FDPTU.privacy.dp_trainer import DPClientTrainer
from FDPTU.federated.server import FederatedServer
from FDPTU.metrics.metric_logger import MetricLogger

# =========================
# CONFIG
# =========================
DATA_ROOT = "/content/drive/MyDrive/datasets/face_attack"
NUM_CLIENTS = 45
BATCH_SIZE = 32
ROUNDS = 10                 # keep small for DP
CLIENT_FRACTION = 0.2
LR = 1e-3

# DP PARAMETERS
EPSILON = 5.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

client_loaders = {}

for i in range(1, NUM_CLIENTS + 1):
    cid = f"{i}"

    ds = FaceAttackDataset(
        root_dir=DATA_ROOT,
        client_id=cid,
        transform=face_transform
    )

    client_loaders[cid] = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

def trainer_fn(model, loader):
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    return DPClientTrainer(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        device=DEVICE,
        max_grad_norm=MAX_GRAD_NORM
    )

metric_logger = MetricLogger(
    "/content/drive/MyDrive/FDPTU/results/face_dpfedavg_metrics.json"
)

server = FederatedServer(
    global_model=FaceClientModel().to(DEVICE),
    client_loaders=client_loaders,
    client_trainer_fn=trainer_fn,
    rounds=ROUNDS,
    client_fraction=CLIENT_FRACTION,
    metric_logger=metric_logger,
    val_loader=client_loaders["1"],
    method_name="DP-FedAvg-Face"
)

server.train()
torch.save(
    server.global_model.state_dict(),
    "/content/drive/MyDrive/FDPTU/checkpoints/dpfedavg_face.pt"
)