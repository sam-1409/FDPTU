import torch.nn as nn
import torch
from FDPTU.models.multimodal_encoder import MultimodalEncoder

class MultimodalClientModel(nn.Module):

    def __init__(self, embed_dim=128, fused_dim=256, num_classes=2):
        super().__init__()
        self.encoder = MultimodalEncoder(embed_dim, fused_dim)
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, iris, fingerprint):
        fused_feat = self.encoder(iris, fingerprint)
        return self.classifier(fused_feat)

    def extract_embedding(self, iris, fingerprint):
        self.eval()
        with torch.no_grad():
            emb = self.encoder(iris, fingerprint)
        return emb