import numpy as np

def mock_mia_baseline(name, base_auc):
    np.random.seed(42)

    fpr = np.linspace(0, 1, 100)
    tpr = fpr + np.random.normal(0, 0.03, size=fpr.shape)
    tpr = np.clip(tpr, 0, 1)

    return {
        "auc": base_auc,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "note": f"{name} (simulated baseline)"
    }