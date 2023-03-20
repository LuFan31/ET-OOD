from pathlib import Path

from torch.utils.data import DataLoader
import torch
import numpy as np
import random

from .imagename_dataset import ImagenameDataset


def get_dataset(
    root_dir: str = "data",
    benchmark: str = "cifar10",
    num_classes: int = 10,
    name: str = "cifar10",
    stage: str = "train",
    interpolation: str = "bilinear",
):
    root_dir = Path(root_dir)
    data_dir = root_dir / "images"  
    imglist_dir = root_dir / "imglist" / f"benchmark_{benchmark}"  

    return ImagenameDataset(
        name=name,
        stage=stage,
        interpolation=interpolation,
        imglist=imglist_dir / f"{stage}_{name}.txt", 
        root=data_dir,   
        num_classes=num_classes,
    )


def get_dataloader(
    root_dir: str = "data",
    benchmark: str = "cifar10",
    num_classes: int = 10,
    name: str = "cifar10",
    stage: str = "train",
    interpolation: str = "bilinear",
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
):
    dataset = get_dataset(root_dir, benchmark, num_classes, name, stage, interpolation)

    # def seed_worker(worker_id):
    #     worker_seed = torch.initial_seed() % 2 ** 32
    #     np.random.seed(worker_seed)
    #     random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def get_ext_dataloader(
    root_dir: str = "data",
    benchmark: str = "cifar10",
    num_classes: int = 10,
    name: str = "cifar10",
    stage: str = "train",
    interpolation: str = "bilinear",
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
):
    dataset = get_dataset(root_dir, benchmark, num_classes, name, stage, interpolation)

    return DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )