import random

def sample_clients(client_ids, fraction=0.3, seed=None):
    if seed is not None:
        random.seed(seed)

    k = max(1, int(len(client_ids) * fraction))
    return random.sample(client_ids, k)