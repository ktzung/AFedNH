import os
import numpy as np
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset

class ICH(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        assert self.mode in ["train", "test"]
        self.transform = transform

        csv_file = os.path.join(self.root, self.mode+"_ICH.csv")
        self.file = pd.read_csv(csv_file)
        self.images = self.file["id"].values
        self.labels = self.file.iloc[:, 1:].values.astype("int")
        self.targets = np.argmax(self.labels, axis=1)
        self.n_classes = len(np.unique(self.targets))
        assert self.n_classes == 5

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        id, target = self.images[index], self.targets[index]
        img = self.read_image(id)

        img = self.transform(img)
        return img, target

    def read_image(self, id):
        image_path = os.path.join(self.root, "ich_images", id+".png")
        image = Image.open(image_path).convert("RGB")
        return image