from collections import deque

def bfs_affected_rounds(
    target_client,
    shard_manager,
    round_logs
):

    target_shard = shard_manager.get_shard(target_client)

    affected_rounds = set()
    visited_shards = set()
    queue = deque([target_shard])

    while queue:
        shard = queue.popleft()
        if shard in visited_shards:
            continue

        visited_shards.add(shard)

        for rnd, clients in round_logs.items():
            for c in clients:
                if shard_manager.get_shard(c) == shard:
                    affected_rounds.add(rnd)

    return sorted(affected_rounds)