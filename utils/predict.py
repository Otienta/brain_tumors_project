import argparse
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
from models.cnn import CNN1
from models.cnn_tf import create_cnn_model
from utils.prep import preprocess_image

def predict_image(image_path, model_type, model_name):
    class_names = ['Glioma', 'Meningioma', 'Notumor', 'Pituitary']
    image = Image.open(image_path).convert('RGB')
    
    if model_type == 'pytorch':
        model = CNN1(num_classes=4)
        model.load_state_dict(torch.load(f"{model_name}_model.torch", map_location='cpu'))
        model.eval()
        image = preprocess_image(image, framework='pytorch')
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()
    else:
        model = tf.keras.models.load_model(f"{model_name}_model.h5")
        image = preprocess_image(image, framework='tensorflow')
        output = model.predict(image)
        prediction = np.argmax(output, axis=1)[0]
    
    return class_names[prediction]

def main():
    parser = argparse.ArgumentParser(description="Predict brain tumor class")
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--model', type=str, choices=['pytorch', 'tensorflow'], default='pytorch')
    parser.add_argument('--model_name', type=str, default='ousmane')
    args = parser.parse_args()
    
    prediction = predict_image(args.image, args.model, args.model_name)
    print(f"Predicted class: {prediction}")

if __name__ == '__main__':
    main()