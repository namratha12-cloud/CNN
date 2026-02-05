"""
Evaluation script for CIFAR-10 CNN
"""
import os
import torch
from tqdm import tqdm

import config
from model import get_model
from data_loader import get_data_loaders
from utils import (
    load_checkpoint, plot_confusion_matrix,
    print_classification_report, visualize_predictions
)


def evaluate():
    """
    Evaluate the trained model
    """
    # Create plots directory
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    # Get data loaders
    print("Loading CIFAR-10 dataset...")
    _, test_loader = get_data_loaders()
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"\nLoading model from {config.BEST_MODEL_PATH}")
    model = get_model(num_classes=config.NUM_CLASSES, device=config.DEVICE)
    
    # Load checkpoint
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"Error: Model checkpoint not found at {config.BEST_MODEL_PATH}")
        print("Please train the model first using train.py")
        return
    
    epoch, accuracy = load_checkpoint(model, None, config.BEST_MODEL_PATH)
    print(f"Loaded model from epoch {epoch + 1} with accuracy: {accuracy:.2f}%")
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Statistics
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'acc': f'{100. * correct / total:.2f}%'})
    
    # Calculate final accuracy
    final_accuracy = 100. * correct / total
    
    # Print results
    print("\n" + "=" * 80)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")
    print(f"Correct predictions: {correct}/{total}")
    print("=" * 80)
    
    # Print classification report
    print_classification_report(all_labels, all_predictions)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm_path = os.path.join(config.PLOTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_predictions, cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # Visualize predictions
    print("\nGenerating prediction visualizations...")
    visualize_predictions(model, test_loader, config.DEVICE, num_images=16)
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    evaluate()
