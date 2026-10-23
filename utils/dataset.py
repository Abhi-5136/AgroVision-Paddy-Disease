import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MultimodalPaddyDataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, is_train: bool = True):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if is_train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Weather sequence shape: [7, 7] (7 days x 7 features)
        weather_seq = row.filter(like="day_").values.astype(np.float32).reshape(7, 7)
        weather_tensor = torch.tensor(weather_seq, dtype=torch.float32)

        label = torch.tensor(row["label_idx"], dtype=torch.long)
        return image, weather_tensor, label