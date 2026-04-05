class ShardManager:

    def __init__(self, clients, shards=5):
        self.shards = {i: [] for i in range(shards)}

        for idx, client in enumerate(sorted(clients)):
            self.shards[idx % shards].append(client)

        self.client_to_shard = {
            client: shard
            for shard, clients in self.shards.items()
            for client in clients
        }

    def get_shard(self, client_id):
        return self.client_to_shard[client_id]

    def get_clients_in_shard(self, shard_id):
        return self.shards[shard_id]