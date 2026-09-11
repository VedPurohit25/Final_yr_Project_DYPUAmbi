import os
import pickle
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

IMAGE_DIR = 'images'

# Load ResNet50
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
model = torch.nn.Sequential(
    *list(resnet.children())[:-2],
    torch.nn.AdaptiveMaxPool2d((1, 1)),
    torch.nn.Flatten()
)
model.eval()

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
    tensor_img = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        result = model(tensor_img).squeeze().numpy()
        
    return result / norm(result)

valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')

# Convert Windows backslashes (\) to Unix forward slashes (/)
filenames = [
    os.path.join(IMAGE_DIR, file).replace('\\', '/')
    for file in os.listdir(IMAGE_DIR) 
    if file.lower().endswith(valid_extensions)
]

feature_list = []
for file in tqdm(filenames, desc="Extracting Features"):
    feature_list.append(extract_features(file, model))

pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

print("Updated embeddings.pkl and filenames.pkl with cross-platform paths!")
