"""
Training script for CIFAR-10 CNN
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from model import get_model, count_parameters
from data_loader import get_data_loaders
from utils import save_checkpoint, load_checkpoint, plot_training_history


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device):
    """
    Validate the model
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Validation')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train():
    """
    Main training function
    """
    # Create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    # Get data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"\nCreating model on device: {config.DEVICE}")
    model = get_model(num_classes=config.NUM_CLASSES, device=config.DEVICE)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = None
    if config.USE_SCHEDULER:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.SCHEDULER_STEP_SIZE,
            gamma=config.SCHEDULER_GAMMA
        )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_acc = 0.0
    start_epoch = 0
    
    # Training loop
    print(f"\nStarting training for {config.EPOCHS} epochs...")
    for epoch in range(start_epoch, config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, test_loader, criterion, config.DEVICE
        )
        
        # Update learning rate
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Learning rate: {current_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                config.BEST_MODEL_PATH
            )
            print(f"✓ Best model saved with accuracy: {best_acc:.2f}%")
        
        # Save last model
        save_checkpoint(
            model, optimizer, epoch, val_acc,
            config.LAST_MODEL_PATH
        )
        
        # Plot training history
        plot_training_history(history, config.PLOTS_DIR)
    
    print("\n" + "=" * 50)
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    train()
