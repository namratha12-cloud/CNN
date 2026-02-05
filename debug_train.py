"""
Debug training script using dummy data to test the pipeline without downloading CIFAR-10
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config
from model import get_model, count_parameters
from utils import save_checkpoint, plot_training_history

def get_dummy_data_loaders():
    """Create dummy data loaders for testing"""
    # Create random images (32x32) and labels (0-9)
    train_size = 100
    test_size = 20
    
    train_images = torch.randn(train_size, 3, 32, 32)
    train_labels = torch.randint(0, 10, (train_size,))
    
    test_images = torch.randn(test_size, 3, 32, 32)
    test_labels = torch.randint(0, 10, (test_size,))
    
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

def debug_train():
    """Debug training function"""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    print("Creating dummy data loaders...")
    train_loader, test_loader = get_dummy_data_loaders()
    
    print(f"Creating model on {config.DEVICE}...")
    model = get_model(num_classes=config.NUM_CLASSES, device=config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("Starting debug training for 2 epochs...")
    for epoch in range(2):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(train_loss) # Just use train loss for dummy validation
        history['val_acc'].append(train_acc)
        
        print(f"Epoch {epoch+1}: Loss {train_loss:.4f}, Acc {train_acc:.2f}%")
        
        # Save "best" model for app testing
        save_checkpoint(model, optimizer, epoch, train_acc, config.BEST_MODEL_PATH)
        plot_training_history(history, config.PLOTS_DIR)

    print("\nDebug training complete. 'best_model.pth' created for testing the web app.")

if __name__ == "__main__":
    debug_train()
