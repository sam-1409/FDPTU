import torch
from torch.utils.data import DataLoader

from FDPTU.datasets.iris_fp_dataset import IrisFingerprintDataset
from FDPTU.datasets.transforms import iris_fp_transform
from FDPTU.models.multimodal_client_model import MultimodalClientModel
from FDPTU.privacy.dp_trainer import DPClientTrainer
from FDPTU.federated.client import ClientTrainer
from FDPTU.unlearning.sfu import sharded_federated_unlearning
from FDPTU.unlearning.shard_manager import ShardManager
from FDPTU.federated.server import FederatedServer
from FDPTU.metrics.metric_logger import MetricLogger

DATA_ROOT = "/content/drive/MyDrive/datasets/iris_fingerprint"
NUM_CLIENTS = 45
BATCH_SIZE = 32
ROUNDS = 10
CLIENT_FRACTION = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR = 1e-3
MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIER = 1.0

client_loaders = {}

for i in range(1, NUM_CLIENTS + 1):
    cid = f"{i}"
    ds = IrisFingerprintDataset(
        DATA_ROOT,
        cid,
        transform=iris_fp_transform()
    )
    client_loaders[cid] = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

clients = list(client_loaders.keys())
shard_mgr = ShardManager(clients, shards=5)

def trainer_fn(model, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    return DPClientTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        max_grad_norm=MAX_GRAD_NORM,
        noise_multiplier=NOISE_MULTIPLIER,
        device=DEVICE
    )

metric_logger = MetricLogger(
    "/content/drive/MyDrive/FDPTU/results/FDPTU_metrics.json"
)

server = FederatedServer(
    global_model=MultimodalClientModel().to(DEVICE),
    client_loaders=client_loaders,
    client_trainer_fn=trainer_fn,
    rounds=ROUNDS,
    client_fraction=CLIENT_FRACTION,
    metric_logger=metric_logger,
    val_loader=client_loaders["1"],
    method_name="FDPTU"
)

server.train()
torch.save(
    server.global_model.state_dict(),
    "/content/drive/MyDrive/FDPTU/checkpoints/FDPTU_before.pt"
)

TARGET_CLIENT = "1"

def retrain_fn(model, affected_clients):
    """
    Retrain affected clients WITHOUT DP.
    DP was already applied in the original training phase.
    """
    model.train()

    for cid in affected_clients:
        loader = client_loaders[f"{int(cid)}"]

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9
        )

        trainer = ClientTrainer(model, optimizer)
        trainer.train_one_epoch(loader)

# Perform SFU
sharded_federated_unlearning(
    server=server,
    target_client=TARGET_CLIENT,
    shard_manager=shard_mgr,
    retrain_fn=retrain_fn
)
torch.save(
    server.global_model.state_dict(),
    "/content/drive/MyDrive/FDPTU/checkpoints/FDPTU_after.pt"
)

print("FDPTU unlearning completed.")