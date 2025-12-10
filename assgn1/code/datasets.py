# datasets.py
import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import torch

def default_transform(size=112):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

class ImageFolderWithPaths(ImageFolder):
    """Extends ImageFolder to return (image, label, path)"""
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return sample, target, path

class TripletImageDataset(Dataset):
    """
    Build triplets on the fly. Expects dataset arranged as ImageFolder (class subfolders).
    For each anchor, sample a positive from same class and a negative from different class.
    Optionally does online hard mining inside batch in training loop too.
    """
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root, transform=transform if transform else default_transform())
        self.transform = self.dataset.transform
        self.samples = self.dataset.samples
        self.class_to_indices = {}
        for idx, (path, class_idx) in enumerate(self.samples):
            self.class_to_indices.setdefault(class_idx, []).append(idx)
        self.indices = list(range(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        anchor_path, anchor_label = self.samples[index]
        anchor = self.transform(Image.open(anchor_path).convert('RGB'))
        # positive
        pos_idx = index
        while pos_idx == index:
            pos_idx = random.choice(self.class_to_indices[anchor_label])
        pos_path, _ = self.samples[pos_idx]
        positive = self.transform(Image.open(pos_path).convert('RGB'))
        # negative
        neg_label = anchor_label
        while neg_label == anchor_label:
            neg_label = random.choice(list(self.class_to_indices.keys()))
        neg_idx = random.choice(self.class_to_indices[neg_label])
        neg_path, _ = self.samples[neg_idx]
        negative = self.transform(Image.open(neg_path).convert('RGB'))
        return anchor, positive, negative, anchor_label

def make_triplet_loader(root, batch_size=32, num_workers=4):
    ds = TripletImageDataset(root=os.path.join(root, 'train'), transform=default_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader

def make_arcface_loader(root, batch_size=32, num_workers=4):
    ds = ImageFolderWithPaths(root=os.path.join(root, 'train'), transform=default_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader

def make_eval_loader(root, batch_size=32, num_workers=4):
    ds = ImageFolderWithPaths(root=os.path.join(root, 'test'), transform=default_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader