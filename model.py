"""
CNN Model Architecture for CIFAR-10 Classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10 classification
    
    Architecture:
    - 3 Convolutional blocks with batch normalization and max pooling
    - 2 Fully connected layers with dropout
    - Output layer with 10 classes
    """
    
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.2)
        
        # Convolutional Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.3)
        
        # Convolutional Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(0.4)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x


def get_model(num_classes=10, device='cpu'):
    """
    Create and return the CNN model
    
    Args:
        num_classes (int): Number of output classes
        device (str or torch.device): Device to load the model on
        
    Returns:
        CIFAR10CNN: The CNN model
    """
    model = CIFAR10CNN(num_classes=num_classes)
    model = model.to(device)
    return model


def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
