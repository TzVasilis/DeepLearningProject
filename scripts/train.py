import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from metrics import calculate_metrics, calculate_pixel_accuracy
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


def train_and_validate(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    NUM_CLASSES: int,
    EPOCHS: int,
    DEVICE: str,
    patience: int,
    use_scheduler: bool = True,
) -> Dict[str, float]:
    """
    Train and validate the model.

    Args:
        model (torch.nn.Module): The neural network model to be trained 
        and validated.
        train_loader (torch.utils.data.DataLoader): Data loader for 
        training data.
        val_loader (torch.utils.data.DataLoader): Data loader for 
        validation data.
        criterion (torch.nn.Module): Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimization algorithm used 
        for updating model weights.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate 
        scheduler.
        NUM_CLASSES (int): Number of classes in the task.
        EPOCHS (int): Number of training epochs.
        DEVICE (str): The device (e.g., 'cuda' or 'cpu') to run the 
        training on.
        patience (int): Number of consecutive epochs with no improvement 
        in validation loss to wait before early stopping.
        use_scheduler (bool, optional): Flag to indicate whether to use 
        the scheduler. Default is True.

    Returns:
        Dict[str, float]: A dictionary containing training and v
        alidation statistics.
    """

    train_losses = []
    val_losses = []
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []

    since = time.time()
    best_val_loss = float("inf")
    current_patience = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        class_iou = {class_id: 0.0 for class_id in range(NUM_CLASSES)}
        class_dice = {class_id: 0.0 for class_id in range(NUM_CLASSES)}
        for batch in train_loader:
            images, masks = batch
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            masks = masks.long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if use_scheduler:
            scheduler.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()

        val_loss = 0
        iou_score = 0
        dice_score = 0
        pixel_accuracy = 0

        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                masks = masks.long()

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Calculate IoU and Dice.
                iou, dice = calculate_metrics(outputs, masks, NUM_CLASSES)
                iou_score += iou.mean()
                dice_score += dice.mean()

                # Calculate pixel accuracy.
                pixel_accuracy += calculate_pixel_accuracy(outputs, masks)

                # Calculate class-wise IoU and Dice.
                for class_id in range(NUM_CLASSES):
                    class_iou[class_id] += iou[class_id]
                    class_dice[class_id] += dice[class_id]

        # For DeepLabV3+.
        # if use_scheduler:
        #     scheduler.step(val_loss)

        val_loss /= len(val_loader)
        iou_score /= len(val_loader)
        dice_score /= len(val_loader)
        pixel_accuracy /= len(val_loader)

        val_losses.append(val_loss)
        iou_scores.append(iou_score)
        dice_scores.append(dice_score)
        pixel_accuracies.append(pixel_accuracy)

        print(
            "========================================================================================================================================"
        )
        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, IoU: {iou_score}, Dice: {dice_score}, Pixel Accuracy: {pixel_accuracy}"
        )

        for class_id in range(NUM_CLASSES):
            class_iou[class_id] = np.clip(
                class_iou[class_id] / len(val_loader), 0, 1
            )
            class_dice[class_id] = np.clip(
                class_dice[class_id] / len(val_loader), 0, 1
            )
            print(
                f"Class {class_id} - IoU: {class_iou[class_id]}, 
                Dice: {class_dice[class_id]}"
            )

        # Check for early stopping.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            current_patience = 0
        else:
            current_patience += 1

        if current_patience >= patience:
            print(
                f"Early stopping at epoch {epoch+1} with best validation 
                loss: {best_val_loss:.4f}"
            )
            break

    time_elapsed = time.time() - since
    print(
        "Training time: {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "iou_scores": iou_scores,
        "dice_scores": dice_scores,
        "class_iou": class_iou,
        "class_dice": class_dice,
        "pixel_accuracies": pixel_accuracies,
    }
