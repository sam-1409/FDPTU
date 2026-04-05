import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_global_model(model, dataloader, modality, device="cpu"):
    model.eval()

    if "Face" in modality or "face" in modality:
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                logits = model(images)
                preds = torch.argmax(logits, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }

    norms = []
    similarities = []

    with torch.no_grad():
        for batch in dataloader:
            iris = batch["iris"].to(device)
            fp = batch["fingerprint"].to(device)

            emb = model.extract_embedding(iris, fp)
            emb = F.normalize(emb, dim=1)

            norms.extend(torch.norm(emb, dim=1).cpu().numpy())
            similarities.append(torch.mm(emb, emb.t()).mean().item())

    return {
        "embedding_norm_mean": float(np.mean(norms)),
        "embedding_norm_std": float(np.std(norms)),
        "intra_batch_similarity": float(np.mean(similarities))
    }