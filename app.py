"""
Flask web application for CIFAR-10 image classification
"""
import os
import io
import base64
import torch
from PIL import Image
from flask import Flask, render_template, request, jsonify
import torchvision.transforms as transforms
import numpy as np

import config
from model import get_model
from utils import load_checkpoint

app = Flask(__name__)

# Global model variable
model = None


def load_model():
    """Load the trained model"""
    global model
    
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"Warning: Model checkpoint not found at {config.BEST_MODEL_PATH}")
        return False
    
    model = get_model(num_classes=config.NUM_CLASSES, device=config.DEVICE)
    epoch, accuracy = load_checkpoint(model, None, config.BEST_MODEL_PATH)
    model.eval()
    
    print(f"Model loaded from epoch {epoch + 1} with accuracy: {accuracy:.2f}%")
    return True


def preprocess_image(image):
    """
    Preprocess image for model prediction
    
    Args:
        image: PIL Image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    return transform(image).unsqueeze(0)


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', class_names=config.CLASS_NAMES)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get image from request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and preprocess image
        image = Image.open(file.stream).convert('RGB')
        input_tensor = preprocess_image(image).to(config.DEVICE)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
        
        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        top5_predictions = [
            {
                'class': config.CLASS_NAMES[idx],
                'probability': float(prob * 100)
            }
            for idx, prob in zip(top5_idx.cpu().numpy(), top5_prob.cpu().numpy())
        ]
        
        # Prepare response
        response = {
            'predicted_class': config.CLASS_NAMES[predicted.item()],
            'confidence': float(confidence.item() * 100),
            'top5_predictions': top5_predictions
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/random_sample', methods=['GET'])
def random_sample():
    """Get a random sample from CIFAR-10 test set or generate dummy if missing"""
    try:
        from data_loader import get_data_loaders
        # Check if dataset exists
        dataset_path = os.path.join(config.DATA_DIR, 'cifar-10-batches-py')
        
        if os.path.exists(dataset_path):
            _, test_loader = get_data_loaders()
            dataset = test_loader.dataset
            idx = np.random.randint(0, len(dataset))
            image, label = dataset[idx]
            
            # Denormalize image
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
            image_denorm = image * std + mean
            image_denorm = torch.clamp(image_denorm, 0, 1)
            
            # Convert to PIL Image
            image_np = (image_denorm.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            label_name = config.CLASS_NAMES[label]
        else:
            # Generate dummy image for demonstration
            image_np = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            label_name = "Dummy Sample (Dataset still downloading)"
            
        pil_image = Image.fromarray(image_np)
        
        # Convert to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'image': f'data:image/png;base64,{img_str}',
            'true_label': label_name
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load model
    if load_model():
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please train the model first using train.py")
