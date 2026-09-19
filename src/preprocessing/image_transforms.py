"""Image transform pipelines for Computer Vision."""

from typing import Tuple
import torchvision.transforms as transforms


def get_train_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Build training transforms with robust data augmentation."""
    return transforms.Compose([
        transforms.Resize((int(image_size[0] * 1.14), int(image_size[1] * 1.14))),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_test_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Build validation and test transforms (deterministic evaluation)."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
