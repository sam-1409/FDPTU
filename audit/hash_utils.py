import hashlib
import torch

def hash_state_dict(state_dict):
    """
    Generates a SHA-256 hash of a model state_dict
    """
    hasher = hashlib.sha256()

    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if tensor.dtype.is_floating_point:
            hasher.update(tensor.cpu().numpy().tobytes())
        else:
            hasher.update(str(tensor).encode())

    return hasher.hexdigest()