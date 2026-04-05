import copy
from FDPTU.unlearning.bfs_traversal import bfs_affected_rounds

from FDPTU.audit.hash_utils import hash_state_dict
from FDPTU.audit.proof import ProofOfRemoval
from FDPTU.audit.audit_logger import AuditLogger

def sharded_federated_unlearning(
    server,
    target_client,
    shard_manager,
    retrain_fn
):
    before_state = copy.deepcopy(server.global_model.state_dict())
    affected_rounds = bfs_affected_rounds(
        target_client,
        shard_manager,
        server.round_logs
    )

    if not affected_rounds:
        print("No affected rounds found.")
        return server.global_model

    rollback_round = min(affected_rounds) - 1

    if rollback_round == 0:
        print("Rolling back to initial model state")
        base_state = server.initial_state
    else:
        print(f"Rolling back to round {rollback_round}")
        base_state = server.checkpoints.get(rollback_round)

    server.global_model.load_state_dict(base_state)

    affected_clients = set()
    for rnd in affected_rounds:
        affected_clients.update(server.round_logs[rnd])

    affected_clients = [
        c for c in affected_clients
        if c != target_client
    ]

    print("Retraining clients:", affected_clients)

    retrain_fn(
        server.global_model,
        affected_clients
    )


    pre_hash = hash_state_dict(before_state)
    post_hash = hash_state_dict(server.global_model.state_dict())

    proof = ProofOfRemoval(
        client_id=target_client,
        shard_id=shard_manager.get_shard(target_client),
        affected_rounds=affected_rounds,
        pre_hash=pre_hash,
        post_hash=post_hash
    )

    audit_logger = AuditLogger(
        log_path="/content/drive/MyDrive/FDPTU/audit/unlearning_log.json"
    )

    audit_logger.log(proof)

    print("Proof of removal generated and logged.")