import streamlit as st
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys

# Ajouter le répertoire racine au chemin de recherche
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)
print("Project root added to sys.path:", project_root)
print("sys.path:", sys.path)

from models.cnn import CustomCNN
from utils.prep import get_pytorch_transforms
from models.cnn_tf import create_cnn_model

# Charger les modèles
@st.cache_resource
def load_pytorch_model():
    try:
        pytorch_model = CustomCNN(num_classes=4)
        pytorch_model.load_state_dict(torch.load(os.path.join('models', 'model.pth'), map_location='cpu'))
        pytorch_model.eval()
        return pytorch_model
    except Exception as e:
        st.error(f"Erreur PyTorch : {e}")
        return None

@st.cache_resource
def load_tf_model():
    try:
        tf_model = tf.keras.models.load_model(os.path.join('models', 'model_tf.h5'))
        return tf_model
    except Exception as e:
        st.error(f"Erreur TensorFlow : {e}")
        return None

pytorch_model = load_pytorch_model()
tf_model = load_tf_model()

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title("Détection de tumeurs cérébrales")
model_type = st.selectbox("Choisir le modèle", ["PyTorch", "TensorFlow"])
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Image téléchargée", use_column_width=True)

    transform = get_pytorch_transforms()[1]  # test_transforms
    if st.button("Prédire"):
        try:
            if model_type == "PyTorch" and pytorch_model is not None:
                image_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    output = pytorch_model(image_tensor)
                    prediction = torch.argmax(output, dim=1).item()
                    st.write(f"Prédiction : {class_names[prediction]}")
            elif model_type == "TensorFlow" and tf_model is not None:
                image_array = np.array(image.resize((224, 224))) / 255.0
                image_array = np.expand_dims(image_array, axis=0)
                output = tf_model.predict(image_array)
                prediction = np.argmax(output, axis=1)[0]
                st.write(f"Prédiction : {class_names[prediction]}")
            else:
                st.error("Modèle non chargé.")
        except Exception as e:
            st.error(f"Échec de la prédiction : {str(e)}")