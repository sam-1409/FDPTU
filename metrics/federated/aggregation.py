import torch

def fedavg(state_dicts):
    """
    Federated Averaging with safe dtype handling
    """
    avg_state = {}

    for key in state_dicts[0].keys():
        tensors = [sd[key] for sd in state_dicts]

        # If tensor is floating point → average
        if tensors[0].dtype in (torch.float32, torch.float64):
            avg_state[key] = torch.mean(
                torch.stack(tensors), dim=0
            )
        else:
            # For int / bool / counters (e.g., BatchNorm buffers)
            avg_state[key] = tensors[0].clone()

    return avg_state