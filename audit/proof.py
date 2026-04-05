import time

class ProofOfRemoval:
    """
    Cryptographic proof that a client's contribution was removed
    """

    def __init__(
        self,
        client_id,
        shard_id,
        affected_rounds,
        pre_hash,
        post_hash
    ):
        self.client_id = client_id
        self.shard_id = shard_id
        self.affected_rounds = affected_rounds
        self.pre_hash = pre_hash
        self.post_hash = post_hash
        self.timestamp = time.time()

    def to_dict(self):
        return {
            "client_id": self.client_id,
            "shard_id": self.shard_id,
            "affected_rounds": self.affected_rounds,
            "pre_model_hash": self.pre_hash,
            "post_model_hash": self.post_hash,
            "timestamp": self.timestamp
        }