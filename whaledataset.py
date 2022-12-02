import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class WhaleDataset(Dataset):
    """
    Stores segmentation input and label images
    """

    def __init__(self, directory, img_dim=128):
        directory = Path(directory)

        self.image_paths = list(directory.glob("**/*.jpg"))
        self.img_dim = img_dim
        self.transform = transforms.Compose([
            transforms.Resize((self.img_dim, self.img_dim)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        input_img = Image.open(self.image_paths[index])
        input_img = self.transform(input_img)
        return input_img

    def __len__(self):
        return len(self.image_paths)
