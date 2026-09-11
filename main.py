import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Load precomputed embeddings and image file paths
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load PyTorch ResNet50 model
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
model = torch.nn.Sequential(
    *list(resnet.children())[:-2],
    torch.nn.AdaptiveMaxPool2d((1, 1)),
    torch.nn.Flatten()
)
model.eval()

# Define image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

st.title('E-commerce Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        os.makedirs('uploads', exist_ok=True)
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception:
        return 0

def feature_extraction(img_path, model):
    img = Image.open(img_path).convert('RGB')
    tensor_img = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        result = model(tensor_img).squeeze().numpy()
        
    return result / norm(result)

def recommend(features, feature_list):
    # Dynamically scale n_neighbors to match total dataset size up to 5 recommendations
    num_samples = len(feature_list)
    num_neighbors = min(5, num_samples)
    
    neighbors = NearestNeighbors(n_neighbors=num_neighbors, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Streamlit App Execution
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display uploaded image
        display_image = Image.open(uploaded_file)
        st.subheader("Uploaded Query Image")
        st.image(display_image, width=300)
        
        # Extract features
        uploaded_image_path = os.path.join("uploads", uploaded_file.name)
        features = feature_extraction(uploaded_image_path, model)
        
        # Compute recommendations
        indices = recommend(features, feature_list)
        recommended_indices = indices[0]
        
        # Render recommendation results in dynamic grid
        st.subheader("Recommended Products")
        if len(recommended_indices) > 0:
            cols = st.columns(len(recommended_indices))
            for i, idx in enumerate(recommended_indices):
                with cols[i]:
                    st.image(filenames[idx], use_container_width=True)
    else:
        st.error("Some error occurred during file upload. Please try again.")
