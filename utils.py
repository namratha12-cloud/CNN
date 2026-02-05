"""
Utility functions for the CIFAR-10 CNN project
"""
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import config


def save_checkpoint(model, optimizer, epoch, accuracy, filepath):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        accuracy: Current accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint file
        
    Returns:
        tuple: (epoch, accuracy)
    """
    checkpoint = torch.load(filepath, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
    return epoch, accuracy


def plot_training_history(history, save_dir):
    """
    Plot training history
    
    Args:
        history: Dictionary containing training history
        save_dir: Directory to save plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_classification_report(y_true, y_pred):
    """
    Print classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    report = classification_report(
        y_true, y_pred,
        target_names=config.CLASS_NAMES,
        digits=4
    )
    print("\nClassification Report:")
    print("=" * 80)
    print(report)
    print("=" * 80)


def visualize_predictions(model, test_loader, device, num_images=16):
    """
    Visualize model predictions
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to run on
        num_images: Number of images to visualize
    """
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(test_loader))
    images, labels = images[:num_images], labels[:num_images]
    images_device = images.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images_device)
        _, predicted = outputs.max(1)
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx in range(num_images):
        # Denormalize image
        img = images[idx].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Plot
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        true_label = config.CLASS_NAMES[labels[idx]]
        pred_label = config.CLASS_NAMES[predicted[idx].cpu()]
        
        color = 'green' if labels[idx] == predicted[idx].cpu() else 'red'
        axes[idx].set_title(
            f'True: {true_label}\nPred: {pred_label}',
            color=color, fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Predictions visualization saved to {config.PLOTS_DIR}/predictions.png")
