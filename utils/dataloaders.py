import os
import random
from typing import Callable, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(
        self,
        data: List[str],
        transform: Optional[Callable] = None,
        is_train: bool = True,
    ):
        """
        Custom dataset class for loading images and masks with optional 
        transformations.

        Args:
            data (List[str]): List of file paths to the data files.
            transform (Optional[Callable]): Albumentations transform to 
            be applied. Default is None.
            is_train (bool, optional): Flag indicating whether the 
            dataset is for training. Default is True.
        """

        self.data = data
        self.transform = transform
        self.is_train = is_train

        if self.is_train and any("photo_" in file for file in data):
            self.shift_transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=30, p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomResizedCrop(256, 256, scale=(0.8, 1.0), p=0.5),
                ]
            )

            self.color_transforms = A.Compose(
                [
                    A.GaussNoise(var_limit=(0.01, 0.02), p=0.4),
                    A.Blur(blur_limit=(1, 3), p=0.4),
                    A.RandomBrightnessContrast(p=0.4),
                    A.ToGray(p=0.3),
                    A.RandomSnow(
                        snow_point_lower=0.01,
                        snow_point_upper=0.02,
                        brightness_coeff=0.5,
                        p=0.2,
                    ),
                    A.Cutout(num_holes=5, max_h_size=10, max_w_size=10, p=0.3),
                ]
            )

        else:
            pass

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """

        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image tensor and mask 
            tensor.
        """

        input_data = self.data[idx]
        data = np.load(input_data)

        # Extract image and mask.
        image = data[:, :, :3].astype(np.float32) / 255.0
        mask = data[:, :, 3]
        mask = (mask / 10).astype(np.uint8)

        # Transforms.
        if self.is_train and "photo_" in input_data:
            shift_transformed = self.shift_transform(image=image, mask=mask)
            image = shift_transformed["image"]
            mask = shift_transformed["mask"]
            color_transformed = self.color_transforms(image=image)
            image = color_transformed["image"]
        elif not self.is_train:
            pass

        image = transforms.ToTensor()(image)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


def data_loaders(
    model_type: Optional[str] = None,
    train_data: Optional[List[str]] = None,
    test_data: Optional[List[str]] = None,
    BATCH_SIZE: Optional[int] = None,
) -> Union[DataLoader, Tuple[DataLoader]]:
    """
    Create data loaders for training, validation, and testing based on 
    the specified model type.

    Args:
        model_type (Optional[str]): Type of the model. Options: "unet", 
        "unet++", "deeplabv3+", or None.
        train_data (Optional[List[str]]): List of file paths to training 
        data files. Default is None.
        test_data (Optional[List[str]]): List of file paths to testing 
        data files.
        BATCH_SIZE (Optional[int]): Batch size for data loaders. It is 
        determined based on the model_type. Default is None.

    Returns:
        Union[DataLoader, Tuple[DataLoader]]: Data loaders for training, 
        validation, and testing.
    """

    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    file_sets = {}

    if model_type == "unet":
        BATCH_SIZE = 32
    elif model_type == "unet++":
        BATCH_SIZE = 8
    elif model_type == "deeplabv3+":
        BATCH_SIZE = 16
    elif model_type == None:
        BATCH_SIZE = 30

    if train_data is not None:
        for filename in train_data:
            file_sets[os.path.basename(filename)] = "train"

        split_ratio = 0.9
        split_index = int(len(train_data) * split_ratio)
        val_indices = list(range(split_index, len(train_data)))

        np.random.seed(random_seed)
        np.random.shuffle(val_indices)

        for i in val_indices:
            file_sets[os.path.basename(train_data[i])] = "val"

        common_filenames = set(
            filename
            for filename, set_type in file_sets.items()
            if set_type == "train" and file_sets[filename] == "val"
        )

        if common_filenames:
            raise ValueError(
                f"Common files found between training and validation 
                sets: 
                {common_filenames}"
            )

        train_dataset = CustomDataset(train_data, is_train=True)
        val_dataset = CustomDataset(
            [train_data[i] for i in val_indices], is_train=False
        )
        test_dataset = CustomDataset(test_data, is_train=False)

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

        return train_loader, val_loader, test_loader

    else:
        test_dataset = CustomDataset(test_data, is_train=False)
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False
        )
        return test_loader
