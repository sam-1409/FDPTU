def build_client_registry():
    iris_fp_clients = [f"IF_{i:03d}" for i in range(1, 46)]
    face_clients = [f"FACE_{i:03d}" for i in range(1, 46)]

    return {
        "iris_fp": iris_fp_clients,
        "face": face_clients
    }