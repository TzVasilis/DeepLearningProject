from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

# Color mapping for class labels.
label_colors = {
    0: [0, 0, 0],  # Black
    1: [250, 149, 10],  # Orange
    2: [19, 98, 19],  # Dark Green
    3: [249, 249, 10],  # Yellow
    4: [10, 248, 250],  # Cyan
    5: [149, 7, 149],  # Purple
    6: [5, 249, 9],  # Light Green
    7: [20, 19, 249],  # Blue
    8: [249, 9, 250],  # Pink
    9: [0, 0, 0],  # Black
}


def label_to_color_mask(labels: np.ndarray) -> np.ndarray:
    """
    Convert class labels to color masks using a predefined color
    mapping.

    Args:
        labels (numpy.ndarray): Class labels for segmentation.

    Returns:
        numpy.ndarray: Color masks corresponding to class labels.
    """

    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    color_masks = np.zeros(
        (labels.shape[0], labels.shape[1], 3), dtype=np.uint8
    )
    for label_value, color in label_colors.items():
        mask = labels == label_value
        color_masks[mask] = color
    return color_masks


def show_images_and_masks(
    images: torch.Tensor, masks: torch.Tensor, num_samples: int = 4
) -> None:
    """
    Display images and corresponding masks for visualization.

    Args:
        images (torch.Tensor): Input images.
        masks (torch.Tensor): Segmentation masks.
        num_samples (int, optional): Number of samples to visualize.
        Default is 4.
    """

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))

    for i in range(num_samples):
        image, label = images[i].permute(1, 2, 0), masks[i]
        ax1, ax2 = axes[i]

        ax1.imshow(image)
        ax1.set_title("Image")
        ax1.axis("off")

        color_mask = label_to_color_mask(label)
        ax2.imshow(color_mask)
        ax2.set_title("Mask")
        ax2.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=-0.6)
    plt.show()


def plot_training_validation_losses(
    epochs: List[int], train_losses: List[float], val_losses: List[float]
) -> None:
    """
    Plot training and validation loss over epochs.

    Args:
        epochs (List[int]): List of epoch numbers.
        train_losses (List[float]): Training loss values.
        val_losses (List[float]): Validation loss values.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Training Loss", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="x")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_predictions(
    test_images: torch.Tensor,
    actual_labels: torch.Tensor,
    predicted_masks: torch.Tensor,
    num_samples: int = 4,
) -> None:
    """
    Visualize predicted segmentation masks along with input images and
    actual masks.

    Args:
        test_images (torch.Tensor): Input images.
        actual_labels (torch.Tensor): Ground truth segmentation masks.
        predicted_masks (torch.Tensor): Predicted segmentation masks.
        num_samples (int, optional): Number of samples to visualize.
        Default is 4.
    """

    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    test_images = test_images.cpu().numpy()
    predicted_masks = predicted_masks.cpu().numpy()
    actual_labels = actual_labels.cpu().numpy()

    num_samples = min(num_samples, test_images.shape[0])

    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 3 * num_samples))

    for i in range(num_samples):
        image = test_images[i].transpose(1, 2, 0)
        predicted_mask = predicted_masks[i]
        actual_label = actual_labels[i]

        ax1, ax2, ax3 = axes[i]

        ax1.imshow(image)
        ax1.set_title("Image")
        ax1.axis("off")

        color_mask_actual = label_to_color_mask(actual_label)
        ax2.imshow(color_mask_actual)
        ax2.set_title("Actual Mask")
        ax2.axis("off")

        color_mask_predicted = label_to_color_mask(predicted_mask)
        ax3.imshow(color_mask_predicted)
        ax3.set_title("Predicted Mask")
        ax3.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=-0.75, hspace=0.1)
    plt.show()
