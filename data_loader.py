"""
Data loading and preprocessing for CIFAR-10 dataset
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config


def get_transforms(train=True):
    """
    Get data transformations for training or testing
    
    Args:
        train (bool): If True, returns training transforms with augmentation
        
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    if train and config.USE_AUGMENTATION:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=config.RANDOM_CROP_PADDING),
            transforms.RandomHorizontalFlip(p=config.RANDOM_HORIZONTAL_FLIP),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
    
    return transform


def get_data_loaders():
    """
    Create train and test data loaders for CIFAR-10
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Get transforms
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    return train_loader, test_loader


def denormalize(tensor):
    """
    Denormalize a tensor image for visualization
    
    Args:
        tensor: Normalized tensor image
        
    Returns:
        tensor: Denormalized tensor image
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return tensor * std + mean
