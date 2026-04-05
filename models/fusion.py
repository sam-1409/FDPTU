import torch
import torch.nn as nn

class ConcatFusion(nn.Module):
    def __init__(self, iris_dim, fp_dim, fused_dim=256):
        super().__init__()
        self.fc = nn.Linear(iris_dim + fp_dim, fused_dim)

    def forward(self, iris_feat, fp_feat):
        fused = torch.cat([iris_feat, fp_feat], dim=1)
        return self.fc(fused)

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, iris_feat, fp_feat):
        stacked = torch.stack([iris_feat, fp_feat], dim=1)
        weights = self.attn(torch.cat([iris_feat, fp_feat], dim=1))
        weights = weights.unsqueeze(-1)
        return (stacked * weights).sum(dim=1)