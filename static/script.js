// CIFAR-10 Classifier JavaScript

let selectedFile = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewCard = document.getElementById('previewCard');
const previewImage = document.getElementById('previewImage');
const classifyBtn = document.getElementById('classifyBtn');
const randomBtn = document.getElementById('randomBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');

// Upload area click handler
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File input change handler
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
});

// Handle file selection
function handleFile(file) {
    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewCard.style.display = 'block';
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Classify button handler
classifyBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        loadingOverlay.style.display = 'flex';

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        displayResults(data);

    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        loadingOverlay.style.display = 'none';
    }
});

// Random sample button handler
randomBtn.addEventListener('click', async () => {
    try {
        loadingOverlay.style.display = 'flex';

        const response = await fetch('/random_sample');
        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        // Convert base64 to blob
        const blob = await fetch(data.image).then(r => r.blob());
        const file = new File([blob], 'random_sample.png', { type: 'image/png' });

        handleFile(file);

    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        loadingOverlay.style.display = 'none';
    }
});

// Display classification results
function displayResults(data) {
    document.getElementById('predictedClass').textContent = data.predicted_class;
    document.getElementById('confidenceValue').textContent = data.confidence.toFixed(2) + '%';

    // Update confidence badge color based on confidence level
    const badge = document.getElementById('confidenceBadge');
    if (data.confidence >= 80) {
        badge.style.background = 'rgba(79, 172, 254, 0.2)';
        badge.style.borderColor = 'rgba(79, 172, 254, 0.4)';
        badge.style.color = '#4facfe';
    } else if (data.confidence >= 60) {
        badge.style.background = 'rgba(240, 147, 251, 0.2)';
        badge.style.borderColor = 'rgba(240, 147, 251, 0.4)';
        badge.style.color = '#f093fb';
    } else {
        badge.style.background = 'rgba(245, 87, 108, 0.2)';
        badge.style.borderColor = 'rgba(245, 87, 108, 0.4)';
        badge.style.color = '#f5576c';
    }

    // Display top 5 predictions
    const top5Container = document.getElementById('top5Container');
    top5Container.innerHTML = '';

    data.top5_predictions.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.style.animationDelay = `${index * 0.1}s`;

        item.innerHTML = `
            <span class="prediction-item-name">${pred.class}</span>
            <div class="prediction-item-bar">
                <div class="prediction-item-fill" style="width: ${pred.probability}%"></div>
            </div>
            <span class="prediction-item-value">${pred.probability.toFixed(2)}%</span>
        `;

        top5Container.appendChild(item);
    });

    resultsSection.style.display = 'grid';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
