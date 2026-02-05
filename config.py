"""
Configuration file for CIFAR-10 CNN project
"""
import torch

# Data configuration
DATA_DIR = './data'
BATCH_SIZE = 32
NUM_WORKERS = 0  # Set to 0 for better stability on some systems without GPU

# Model configuration
NUM_CLASSES = 10
INPUT_CHANNELS = 3
IMAGE_SIZE = 32
HIDDEN_SIZE = 256
NUM_LAYERS = 2
RNN_DROPOUT = 0.2

# Training configuration
EPOCHS = 5
LEARNING_RATE = 0.01  # Increased slightly for faster convergence in few epochs
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Checkpoint configuration
CHECKPOINT_DIR = './checkpoints'
BEST_MODEL_PATH = './checkpoints/best_model.pth'
LAST_MODEL_PATH = './checkpoints/last_model.pth'

# Visualization configuration
PLOTS_DIR = './plots'

# CIFAR-10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Data augmentation settings
USE_AUGMENTATION = True
RANDOM_CROP_PADDING = 4
RANDOM_HORIZONTAL_FLIP = 0.5

# Learning rate scheduler
USE_SCHEDULER = True
SCHEDULER_STEP_SIZE = 20
SCHEDULER_GAMMA = 0.1
