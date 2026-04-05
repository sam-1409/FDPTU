import torch.nn as nn
from FDPTU.models.face_encoder import FaceEncoder

class FaceClientModel(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.encoder = FaceEncoder(embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat)