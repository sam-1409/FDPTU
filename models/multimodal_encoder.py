import torch.nn as nn
from FDPTU.models.iris_encoder import IrisEncoder
from FDPTU.models.fingerprint_encoder import FingerprintEncoder
from FDPTU.models.fusion import ConcatFusion

class MultimodalEncoder(nn.Module):
    def __init__(self, embed_dim=128, fused_dim=256):
        super().__init__()
        self.iris_encoder = IrisEncoder(embed_dim)
        self.fp_encoder = FingerprintEncoder(embed_dim)
        self.fusion = ConcatFusion(embed_dim, embed_dim, fused_dim)

    def forward(self, iris, fingerprint):
        iris_feat = self.iris_encoder(iris)
        fp_feat = self.fp_encoder(fingerprint)
        return self.fusion(iris_feat, fp_feat)