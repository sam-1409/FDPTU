from FDPTU.models.backbone import SmallCNN

class FaceEncoder(SmallCNN):
    def __init__(self, embedding_dim=128):
        super().__init__(in_channels=3, embedding_dim=embedding_dim)