import torch
from torch.utils.data import DataLoader

from FDPTU.datasets.iris_fp_dataset import IrisFingerprintDataset
from FDPTU.datasets.transforms import iris_fp_transform
from FDPTU.models.multimodal_client_model import MultimodalClientModel
from FDPTU.privacy.dp_trainer import DPClientTrainer
from FDPTU.federated.server import FederatedServer
from FDPTU.metrics.metric_logger import MetricLogger

DATA_ROOT = "<Root directory of the dataset>"

client_loaders = {}
for i in range(1, 46):
    cid = f"IF_{i:03d}"
    ds = IrisFingerprintDataset(DATA_ROOT, cid, transform=iris_fp_transform())
    client_loaders[cid] = DataLoader(ds, batch_size=8, shuffle=True)

def trainer_fn(model, loader):
    return DPClientTrainer(model, loader)

metric_logger = MetricLogger(
    "<Path to the metrics file>"
)

server = FederatedServer(
    global_model=MultimodalClientModel(),
    client_loaders=client_loaders,
    client_trainer_fn=trainer_fn,
    rounds=50,
    client_fraction=0.3,
    metric_logger=metric_logger,
    val_loader=client_loaders["IF_001"],
    method_name="DP-FedAvg"
)

server.train()