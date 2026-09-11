import os
import pickle
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Updated directory path to match the local sample folder
IMAGE_DIR = 'images'

# Load ResNet50 model pretrained on ImageNet
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Replace classification head with AdaptiveMaxPool2d
resnet.fc = torch.nn.Identity()
model = torch.nn.Sequential(
    *list(resnet.children())[:-2],
    torch.nn.AdaptiveMaxPool2d((1, 1)),
    torch.nn.Flatten()
)
model.eval()  # Set model to evaluation mode

# Preprocessing pipeline matching ImageNet standard transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def extract_features(img_path, model):
    img = Image.open(img_path).convert('RGB')
    tensor_img = transform(img).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        result = model(tensor_img).squeeze().numpy()
        
    return result / norm(result)

# Get all valid image file paths
valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
filenames = [
    os.path.join(IMAGE_DIR, file) 
    for file in os.listdir(IMAGE_DIR) 
    if file.lower().endswith(valid_extensions)
]

# Extract features
feature_list = []
for file in tqdm(filenames, desc="Extracting Features"):
    feature_list.append(extract_features(file, model))

# Save pickle files
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

print(f"\nSuccessfully generated embeddings.pkl and filenames.pkl for {len(filenames)} images!")
