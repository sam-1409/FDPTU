import torch

def residual_influence(before_state, after_state):
    diff = 0.0
    total = 0.0

    for key in before_state:
        if before_state[key].dtype.is_floating_point:
            d = torch.norm(
                before_state[key] - after_state[key]
            ).item()
            diff += d
            total += torch.norm(before_state[key]).item()

    return diff / (total + 1e-8)