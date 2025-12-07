import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform_mode=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform_mode = transform_mode
        
        self.resize = transforms.Resize((256, 256))
        self.to_tensor = transforms.ToTensor()
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        
        image = self.resize(image)
        mask = self.resize(mask)
        
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        
        return image, mask

def prepare_data(csv_path, test_size=0.2, val_size=0.1):
    df = pd.read_csv(csv_path, header=0, names=["idxs",'images', 'masks','collages'])
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    val_relative_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=val_relative_size, random_state=42)
    
    return {
        'train': (train_df['images'].tolist(), train_df['masks'].tolist()),
        'val': (val_df['images'].tolist(), val_df['masks'].tolist()),
        'test': (test_df['images'].tolist(), test_df['masks'].tolist())
    }