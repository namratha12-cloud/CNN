"""
RNN Model Architecture for CIFAR-10 Classification
"""
import torch
import torch.nn as nn
import config


class CIFAR10RNN(nn.Module):
    """
    Recurrent Neural Network (LSTM) for CIFAR-10 classification
    
    Architecture:
    - Input sequence: 32 rows of 32x3 pixels (= 96 features per step)
    - Bidirectional LSTM layers
    - Fully connected layer for classification
    """
    
    def __init__(self, input_size=96, hidden_size=256, num_layers=2, num_classes=10):
        super(CIFAR10RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True means input shape is (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=config.RNN_DROPOUT if num_layers > 1 else 0
        )
        
        # Fully Connected Layer
        # * 2 because of bidirectional
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, 3, 32, 32)
        # Convert to: (batch, seq_len=32, input_size=96)
        batch_size = x.size(0)
        
        # Rearrange image rows into a sequence
        # (batch, 3, 32, 32) -> (batch, 32, 3, 32) -> (batch, 32, 96)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, 32, -1)
        
        # LSTM Forward pass
        # out: tensor of shape (batch, seq_len, hidden_size * 2)
        out, _ = self.lstm(x)
        
        # Take the output of the last time step
        out = out[:, -1, :]
        
        # Classification
        out = self.fc(out)
        
        return out


def get_model(num_classes=10, device='cpu'):
    """
    Create and return the RNN model
    
    Args:
        num_classes (int): Number of output classes
        device (str or torch.device): Device to load the model on
        
    Returns:
        CIFAR10RNN: The RNN model
    """
    model = CIFAR10RNN(
        input_size=32*3, 
        hidden_size=config.HIDDEN_SIZE, 
        num_layers=config.NUM_LAYERS, 
        num_classes=num_classes
    )
    model = model.to(device)
    return model


def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
