# CIFAR-10 CNN Image Classifier

An end-to-end deep learning project for classifying CIFAR-10 images using a Convolutional Neural Network (CNN) built with PyTorch. Includes a modern web interface for real-time image classification.

## 🌟 Features

- **Custom CNN Architecture**: 3 convolutional blocks with batch normalization and dropout
- **Complete Training Pipeline**: Automated training with validation, checkpointing, and visualization
- **Comprehensive Evaluation**: Confusion matrix, classification reports, and prediction visualizations
- **Modern Web Interface**: Beautiful Flask web app for real-time image classification
- **CIFAR-10 Dataset**: Automatically downloads and preprocesses the dataset

## 📊 Model Architecture

```
Input (32x32x3)
    ↓
Conv Block 1: Conv2d(64) → BatchNorm → ReLU → Conv2d(64) → BatchNorm → ReLU → MaxPool → Dropout(0.2)
    ↓
Conv Block 2: Conv2d(128) → BatchNorm → ReLU → Conv2d(128) → BatchNorm → ReLU → MaxPool → Dropout(0.3)
    ↓
Conv Block 3: Conv2d(256) → BatchNorm → ReLU → Conv2d(256) → BatchNorm → ReLU → MaxPool → Dropout(0.4)
    ↓
Flatten
    ↓
FC Layer 1: Linear(512) → BatchNorm → ReLU → Dropout(0.5)
    ↓
FC Layer 2: Linear(10)
    ↓
Output (10 classes)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

This will:
- Download the CIFAR-10 dataset automatically
- Train the model for 50 epochs
- Save checkpoints in `./checkpoints/`
- Generate training plots in `./plots/`

### 3. Evaluate the Model

```bash
python evaluate.py
```

This will:
- Load the best model checkpoint
- Evaluate on the test set
- Generate confusion matrix
- Create prediction visualizations

### 4. Run the Web Application

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

## 📁 Project Structure

```
CNN/
├── config.py              # Configuration and hyperparameters
├── data_loader.py         # Data loading and preprocessing
├── model.py               # CNN model architecture
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── utils.py               # Utility functions
├── app.py                 # Flask web application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web interface HTML
├── static/
│   ├── style.css         # Web interface CSS
│   └── script.js         # Web interface JavaScript
├── checkpoints/          # Model checkpoints (created during training)
├── plots/                # Training visualizations (created during training)
└── data/                 # CIFAR-10 dataset (downloaded automatically)
```

## 🎯 CIFAR-10 Classes

The model classifies images into 10 categories:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## ⚙️ Configuration

Edit `config.py` to customize:
- **Training**: epochs, batch size, learning rate
- **Model**: number of classes, architecture parameters
- **Data**: augmentation settings, normalization values
- **Paths**: checkpoint and plot directories

## 📈 Training Details

- **Optimizer**: SGD with momentum (0.9) and weight decay (5e-4)
- **Loss Function**: Cross-Entropy Loss
- **Learning Rate**: 0.001 with step decay
- **Batch Size**: 128
- **Data Augmentation**: Random crop and horizontal flip
- **Regularization**: Batch normalization and dropout

## 🎨 Web Interface Features

- **Drag & Drop**: Upload images via drag-and-drop
- **Random Samples**: Test with random CIFAR-10 images
- **Real-time Classification**: Instant predictions with confidence scores
- **Top-5 Predictions**: View probability distribution
- **Modern UI**: Dark theme with smooth animations

## 📊 Expected Performance

With the default configuration, the model typically achieves:
- **Training Accuracy**: ~90%
- **Validation Accuracy**: ~85%

## 🛠️ Requirements

- Python 3.7+
- PyTorch 2.0+
- torchvision
- Flask
- NumPy
- Matplotlib
- scikit-learn
- Pillow
- tqdm

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📧 Contact

For questions or feedback, please open an issue on the repository.

---

**Built with ❤️ using PyTorch and Flask**
