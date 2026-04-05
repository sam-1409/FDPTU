import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FaceAttackDataset(Dataset):
    """
    Supervised face attack dataset.
    Label: 0 = real, 1 = fake
    """

    def __init__(self, root_dir, client_id, transform=None):
        self.samples = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        client_path = os.path.join(root_dir, client_id)

        for label_name, label in [("real", 0), ("fake", 1)]:
            folder = os.path.join(client_path, label_name)
            for img in os.listdir(folder):
                self.samples.append(
                    (os.path.join(folder, img), label)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "label": label
        }