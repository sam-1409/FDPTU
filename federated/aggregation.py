import torch

def fedavg(state_dicts):
    avg_state = {}
    for key in state_dicts[0].keys():
        tensors = [sd[key] for sd in state_dicts]
        if tensors[0].dtype in (torch.float32, torch.float64):
            avg_state[key] = torch.mean(
                torch.stack(tensors), dim=0
            )
        else:
            avg_state[key] = tensors[0].clone()

    return avg_state