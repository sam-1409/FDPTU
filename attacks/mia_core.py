import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity

@torch.no_grad()
def mia_from_confidence(model, loader, device):
    model.eval()
    scores = []

    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        conf, _ = probs.max(dim=1)
        scores.extend(conf.cpu().numpy())

    return np.array(scores)

@torch.no_grad()
def mia_from_embeddings(model, loader, device):
    model.eval()
    embeddings = []

    for batch in loader:
        iris = batch["iris"].to(device)
        fp = batch["fingerprint"].to(device)

        emb = model.encoder(iris, fp)
        emb = F.normalize(emb, dim=1)
        embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(embeddings)
    ref = embeddings.mean(axis=0, keepdims=True)

    scores = cosine_similarity(embeddings, ref).flatten()
    return scores

def run_mia(model, member_loader, nonmember_loader, device, mode):
    if mode == "confidence":
        m_scores = mia_from_confidence(model, member_loader, device)
        nm_scores = mia_from_confidence(model, nonmember_loader, device)
    elif mode == "embedding":
        m_scores = mia_from_embeddings(model, member_loader, device)
        nm_scores = mia_from_embeddings(model, nonmember_loader, device)
    else:
        raise ValueError("Unknown MIA mode")

    y_true = np.concatenate([np.ones(len(m_scores)), np.zeros(len(nm_scores))])
    y_score = np.concatenate([m_scores, nm_scores])

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    return {
        "auc": float(auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    }