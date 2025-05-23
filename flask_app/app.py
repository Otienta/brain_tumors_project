from flask import Flask, request, jsonify, send_from_directory
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys

# Ajouter le répertoire racine de brain_tumors_project au chemin de recherche
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
print("Project root added to sys.path:", project_root)
print("sys.path:", sys.path)  # Pour déboguer

from models.cnn import CustomCNN
from utils.prep import get_pytorch_transforms
from models.cnn_tf import create_cnn_model

app = Flask(__name__, static_folder='static', template_folder='templates')

pytorch_model = CustomCNN(num_classes=4)
pytorch_model.load_state_dict(torch.load(os.path.join(project_root, 'model.pth'), map_location='cpu'))
pytorch_model.eval()

tf_model = tf.keras.models.load_model(os.path.join(project_root, 'model_tf.h5'))

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Adapter aux noms des dossiers dans dataset/

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'model' not in request.form:
        return jsonify({'error': 'Model type is required'}), 400
    
    model_type = request.form['model']
    if model_type not in ['pytorch', 'tensorflow']:
        return jsonify({'error': 'Invalid model type'}), 400

    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({'error': 'No image file provided'}), 400

    try:
        image = Image.open(request.files['image']).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

    transform = get_pytorch_transforms()[1]  # Utilise test_transforms
    try:
        if model_type == 'pytorch':
            image_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = pytorch_model(image_tensor)
                prediction = torch.argmax(output, dim=1).item()
        else:
            image_array = np.array(image.resize((224, 224))) / 255.0
            image_array = image_array[np.newaxis, ...]
            output = tf_model.predict(image_array)
            prediction = np.argmax(output, axis=1)[0]
        
        return jsonify({'prediction': class_names[prediction]})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)