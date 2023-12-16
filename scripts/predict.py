from typing import Tuple

import torch
from metrics import calculate_metrics, calculate_pixel_accuracy
from plot_utils import visualize_predictions


def make_predictions(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    DEVICE: str,
    NUM_CLASSES: int,
    num_samples: int,
) -> Tuple[float, float, float]:
    """
    Make predictions using a trained model on a test dataset and
    calculate mIoU, mean Dice score, and mean Pixel Accuracy.

    Args:
        model (torch.nn.Module): The trained segmentation model.
        test_loader (torch.utils.data.DataLoader): Data loader for the
        test dataset.
        DEVICE (str): The device (e.g., 'cuda' or 'cpu') to run the
        inference on.
        NUM_CLASSES (int): Number of classes in the segmentation task.
        num_samples (int, optional): Number of samples to visualize.

    Returns:
        Tuple[float, float, float]: Mean IoU (Jaccard) score, mean Dice
        (F1) score, and mean Pixel Accuracy across the test dataset.
    """

    model.eval()

    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    predicted_masks = []
    pixel_accuracy = 0
    pixel_accuracy_score = 0

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            labels = labels.long()

            outputs = model(images)
            predicted_labels = torch.argmax(outputs, dim=1)
            predicted_masks = predicted_labels

            iou, dice = calculate_metrics(
                predicted_labels, labels, NUM_CLASSES
            )
            pixel_accuracy = calculate_pixel_accuracy(predicted_labels, labels)
            pixel_accuracy_score += pixel_accuracy.mean()

    visualize_predictions(
        images, labels, predicted_masks, num_samples=num_samples
    )

    iou_scores.append(iou)
    dice_scores.append(dice)
    pixel_accuracies.append(pixel_accuracy_score)

    return {
        "iou": iou_scores,
        "dice": dice_scores,
        "pixel_accuracy": pixel_accuracies,
    }
