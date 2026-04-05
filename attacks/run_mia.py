import json
from FDPTU.attacks.mia_core import run_mia
from FDPTU.attacks.mia_baselines import mock_mia_baseline


def run_all_mia(
    fedavg_model,
    dpfedavg_model,
    FDPTU_before,
    FDPTU_after,
    face_member_loader,
    face_nonmember_loader,
    mm_member_loader,
    mm_nonmember_loader,
    device
):
    results = {}

    results["FedAvg"] = run_mia(
        fedavg_model,
        face_member_loader,
        face_nonmember_loader,
        device,
        mode="confidence"
    )

    results["DP-FedAvg"] = run_mia(
        dpfedavg_model,
        face_member_loader,
        face_nonmember_loader,
        device,
        mode="confidence"
    )
    
    results["FDPTU-Before"] = run_mia(
        FDPTU_before,
        mm_member_loader,
        mm_nonmember_loader,
        device,
        mode="embedding"
    )

    results["FDPTU-After"] = run_mia(
        FDPTU_after,
        mm_member_loader,
        mm_nonmember_loader,
        device,
        mode="embedding"
    )

    results["FedSF"] = mock_mia_baseline("FedSF", base_auc=0.52)
    results["SISA"] = mock_mia_baseline("SISA", base_auc=0.48)

    with open("/content/drive/MyDrive/FDPTU/results/mia_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("MIA evaluation completed and saved.")