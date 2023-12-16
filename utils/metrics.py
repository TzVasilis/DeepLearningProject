from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, jaccard_score


def calculate_metrics(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate IoU (Jaccard score) and Dice score for a model's
    predictions and target masks.

    Args:
        pred (torch.Tensor): Predicted segmentation masks of shape
        (batch_size, num_classes, height, width).
        target (torch.Tensor): Ground truth masks of shape (batch_size,
        height, width).
        num_classes (int): Number of classes in the segmentation task.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Array of IoU (Jaccard) scores for
        each class and array of Dice (F1) scores for each class.
    """

    # For inference.
    pred = pred.reshape(-1)
    # For training.
    # pred = pred.argmax(dim=1).reshape(-1)
    target = target.reshape(-1)

    iou = jaccard_score(
        target.cpu().numpy(),
        pred.cpu().numpy(),
        average=None,
        labels=range(num_classes),
        zero_division=1,
    )

    dice = f1_score(
        target.cpu().numpy(),
        pred.cpu().numpy(),
        average=None,
        labels=range(num_classes),
        zero_division=1,
    )

    return iou, dice


def calculate_pixel_accuracy(
    predicted_masks: torch.Tensor, actual_masks: torch.Tensor
) -> float:
    """
    Calculate pixel accuracy between predicted segmentation masks and
    ground truth masks.

    Args:
        predicted_masks (torch.Tensor): Predicted segmentation masks
        with shape (batch_size, height, width).
        actual_masks (torch.Tensor): Ground truth segmentation masks
        with shape (batch_size, height, width).

    Returns:
        float: Pixel accuracy, the ratio of correctly predicted pixels
        to total pixels.
    """

    predicted_np = predicted_masks.cpu().numpy()
    actual_np = actual_masks.cpu().numpy()

    correct_pixels = np.sum(predicted_np == actual_np)
    total_pixels = np.prod(actual_np.shape)

    accuracy = correct_pixels / total_pixels

    return accuracy
