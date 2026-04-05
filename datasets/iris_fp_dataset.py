import os
from PIL import Image
from torch.utils.data import Dataset

class IrisFingerprintDataset(Dataset):
    """
    One subject = one federated client
    Modalities:
      - Iris (left + right)
      - Fingerprint (10 fingers)
    """

    def __init__(self, root_dir, client_id, transform=None):
        self.root_dir = root_dir
        self.client_id = client_id
        self.transform = transform

        # Robust client ID parsing
        if "_" in client_id:
            subject_id = client_id.split("_")[-1].lstrip("0")
        else:
            subject_id = client_id.lstrip("0")

        self.subject_path = os.path.join(root_dir, subject_id)
        assert os.path.exists(self.subject_path), f"Client path not found: {self.subject_path}"

        self.iris_images = self._load_iris()
        self.fp_images = self._load_fingerprint()

    def _load_iris(self):
        iris_imgs = []
        for eye in ["left", "right"]:
            eye_path = os.path.join(self.subject_path, eye)
            for fname in sorted(os.listdir(eye_path)):
                if fname.lower().endswith(".bmp"):
                    iris_imgs.append(os.path.join(eye_path, fname))
        return iris_imgs

    def _load_fingerprint(self):
        fp_path = os.path.join(self.subject_path, "Fingerprint")
        return [
            os.path.join(fp_path, f)
            for f in sorted(os.listdir(fp_path))
            if f.lower().endswith(".bmp")
        ]

    def __len__(self):
        return max(len(self.iris_images), len(self.fp_images))

    def __getitem__(self, idx):
        iris_path = self.iris_images[idx % len(self.iris_images)]
        fp_path = self.fp_images[idx % len(self.fp_images)]

        iris_img = Image.open(iris_path).convert("L")
        fp_img = Image.open(fp_path).convert("L")

        if self.transform:
            iris_img = self.transform(iris_img)
            fp_img = self.transform(fp_img)

        return {
            "client_id": self.client_id,
            "iris": iris_img,
            "fingerprint": fp_img
        }