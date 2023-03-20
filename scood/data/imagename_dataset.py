import ast
import io
import logging
import os
from typing import List

import numpy as np
import torch
import torchvision.transforms as trn
from PIL import Image, ImageFile
from torchvision.transforms import InterpolationMode

from .base_dataset import BaseDataset

from .randaugment import RandAugmentMC
from .transforms import RandomErasing
from PIL import ImageFilter
import random

class TwoCropsTransform:

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Convert:
    def __init__(self, mode="RGB"):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


# [mean, std]
dataset_stats = {
    # Training stats
    "cifar10": [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]],
    "cifar100": [[0.507, 0.487, 0.441], [0.267, 0.256, 0.276]],
    "ima100": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    "tin": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    # Testing (open world) stats
    "test": [[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
}


def get_transforms(
    mean: List[float],
    std: List[float],
    stage: str,
    interpolation: str = "bilinear",
):
    interpolation_modes = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
    }
    color_mode = "RGB"

    interpolation = interpolation_modes[interpolation]

    if stage == "train":
        return trn.Compose(
            [
                Convert(color_mode),
                trn.Resize(32, interpolation=interpolation),
                trn.CenterCrop(32),
                trn.RandomHorizontalFlip(),
                trn.RandomCrop(32, padding=int(32*0.125),padding_mode='reflect'),
                trn.ToTensor(),
                trn.Normalize(mean, std),
            ]
        )

    elif stage == "test":
        return trn.Compose(
            [
                Convert(color_mode),
                trn.Resize(32, interpolation=interpolation),
                trn.CenterCrop(32),
                trn.ToTensor(),
                trn.Normalize(mean, std),
            ]
        )

def strong_transforms(
    mean: List[float],
    std: List[float],
    stage: str,
    interpolation: str = "bilinear",):
    interpolation_modes = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
    }
    color_mode = "RGB"

    interpolation = interpolation_modes[interpolation]
    if stage == "train":
        return trn.Compose([
            Convert(color_mode),
            trn.Resize(size=(32, 32), interpolation=interpolation),
            trn.RandomHorizontalFlip(),
            trn.RandomCrop(size=32,padding=int(32*0.125),padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            trn.ToTensor(),
            trn.Normalize(mean, std)
            ])

def OT_transforms(
    mean: List[float],
    std: List[float],
    stage: str,
    interpolation: str = "bilinear",):
    interpolation_modes = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
    }
    color_mode = "RGB"

    interpolation = interpolation_modes[interpolation]
    if stage == "train":
        return trn.Compose([
            Convert(color_mode),
            trn.Resize(32, interpolation=interpolation),
            trn.CenterCrop(32),
            trn.ColorJitter(0.4, 0.4, 0.4, 0.4),
            trn.RandomGrayscale(p=0.2),
            # trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])

# def trip_transforms(
#     mean: List[float],
#     std: List[float],
#     stage: str,
#     interpolation: str = "bilinear",):
#     interpolation_modes = {
#         "nearest": InterpolationMode.NEAREST,
#         "bilinear": InterpolationMode.BILINEAR,
#     }
#     color_mode = "RGB"
#     interpolation = interpolation_modes[interpolation]
#     if stage == "train":
#         return trn.Compose([
#                 Convert(color_mode),
#                 trn.Resize(32, interpolation=interpolation),
#                 trn.RandomHorizontalFlip(p=0.5),
#                 trn.Pad(1),
#                 trn.RandomCrop(32),
#                 trn.ToTensor(),
#                 trn.Normalize(mean, std),
#                 RandomErasing(probability=0.5, mean=mean)
#             ])

def aug_transforms(
    mean: List[float],
    std: List[float],
    stage: str,
    interpolation: str = "bilinear",):
    interpolation_modes = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
    }
    color_mode = "RGB"

    interpolation = interpolation_modes[interpolation]
    if stage == "train":
        augmentation = [
            Convert(color_mode),
            trn.Resize(32, interpolation=interpolation),
            trn.CenterCrop(32),
            trn.RandomHorizontalFlip(), 
            trn.RandomCrop(32, padding=int(32*0.125),padding_mode='reflect'),
            trn.RandomApply([trn.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
            trn.RandomGrayscale(p=0.2),
            # trn.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ]
        # augmentation = [
        #     Convert(color_mode),
        #     trn.Resize(32, interpolation=interpolation),
        #     # trn.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        #     trn.CenterCrop(32),
        #     trn.RandomHorizontalFlip(),
        #     trn.RandomCrop(32, padding=int(32*0.125),padding_mode='reflect'),
        #     trn.ColorJitter(0.4, 0.4, 0.4, 0.4),
        #     trn.RandomGrayscale(p=0.2),
        #     # trn.RandomHorizontalFlip(),
        #     trn.ToTensor(),
        #     trn.Normalize(mean, std),
        # ]
        return TwoCropsTransform(trn.Compose(augmentation))


class ImagenameDataset(BaseDataset):
    def __init__(
        self,
        name,
        stage,
        interpolation,
        imglist,  
        root,  
        num_classes=10,
        maxlen=None,
        dummy_read=False,
        dummy_size=None,
        **kwargs
    ):
        super(ImagenameDataset, self).__init__(**kwargs)

        self.name = name

        with open(imglist) as imgfile:
            self.imglist = imgfile.readlines()
        self.root = root

        mean, std = dataset_stats[stage] if stage == "test" else dataset_stats[name]  
        self.transform_image = get_transforms(mean, std, stage, interpolation)  
        # basic image transformation for online clustering (without augmentations)
        self.transform_aux_image = get_transforms(mean, std, "test", interpolation)
        self.s_transform = strong_transforms(mean, std, "train", interpolation)
        self.OT_transform = OT_transforms(mean, std, "train", interpolation)
        self.aug_transform = aug_transforms(mean, std, "train", interpolation)

        self.num_classes = num_classes
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
        if dummy_read and dummy_size is None:
            raise ValueError("if dummy_read is True, should provide dummy_size")

        self.cluster_id = np.zeros(len(self.imglist), dtype=int)                                      
        self.cluster_reweight = np.ones(len(self.imglist), dtype=float)

        # use pseudo labels for unlabeled dataset during training
        self.pseudo_label = np.array(-1 * np.ones(len(self.imglist)), dtype=int)
        self.ood_conf = np.ones(len(self.imglist), dtype=float)

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)
    ''' 每次传入一个索引index'''
    def getitem(self, index):
        line = self.imglist[index].strip("\n") 
                                               
        tokens = line.split(" ", 1)  
        image_name, extra_str = tokens[0], tokens[1] 
        if self.root != "" and image_name.startswith("/"):
            raise RuntimeError('root not empty but image_name starts with "/"')
        path = os.path.join(self.root, image_name) 
        sample = dict()  
        sample["image_name"] = image_name
        try:
            if not self.dummy_read:  
                with open(path, "rb") as f:
                    content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
            if self.dummy_size is not None:
                sample["data"] = torch.rand(self.dummy_size)
            else: 
                image = Image.open(buff).convert("RGB")
                sample["data"] = self.transform_image(image)
                sample["plain_data"] = self.transform_aux_image(image)  # basic image transformation for online clustering (without augmentations)
                sample["OT_data"] = self.OT_transform(image)
                sample["strong_data"] = self.s_transform(image)
                sample["aug_data"] = self.aug_transform(image)
            extras = ast.literal_eval(extra_str)  
            try:
                for key, value in extras.items(): 
                    sample[key] = value
            except AttributeError:
                sample["label"] = int(extra_str)
            # Generate Soft Label
            soft_label = torch.Tensor(self.num_classes)
            if sample["label"] < 0:  
                soft_label.fill_(1.0 / self.num_classes) 
            else:
                soft_label.fill_(0)
                soft_label[sample["label"]] = 1  
            sample["soft_label"] = soft_label  
            # Deep Clustering Aux Label Assignment for both labeled/unlabeled data
            sample["cluster_id"] = self.cluster_id[index]
            sample["cluster_reweight"] = self.cluster_reweight[index]

            # Deep Clustering Pseudo Label Assignment for unlabeled data
            sample["pseudo_label"] = self.pseudo_label[index]  
            soft_pseudo_label = torch.Tensor(len(sample["soft_label"])) 
            if sample["pseudo_label"] == -1:  
                soft_pseudo_label.fill_(1.0 / len(sample["soft_label"])) 
            else:
                soft_pseudo_label.fill_(0.0)
                soft_pseudo_label[sample["pseudo_label"]] = 1.0
            sample["pseudo_softlabel"] = soft_pseudo_label
            sample["ood_conf"] = self.ood_conf[index]

        except Exception as e:
            logging.error("[{}] broken".format(path))
            raise e
        return sample
