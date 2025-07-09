

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10
num_classes = 2  # fake or real

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
model_save_path = "efficientnet_deepfake.pth"
loaded_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)

# Load the model weights, map them to CPU if necessary
loaded_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))

# If using GPU, move model to the device
loaded_model = loaded_model.to(device)

# Set the model to evaluation mode
loaded_model.eval()

cls=["Fake","Real"]

# Function to predict from an image array
def predict_single_image_from_array(model, img_array, transform):
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray((img_array * 255).astype(np.uint8))  # Assuming img_array is scaled 0 to 1
    # Apply the transform (resize, normalize, etc.)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)

    return predicted.item(),outputs
# Now use the updated function in your face detection loop
def pred_img(img_array):
    # Assuming img_array is the preprocessed NumPy array from the face detection loop
    predicted_class,outputs = predict_single_image_from_array(loaded_model, img_array, transform)
    print(f"Predicted class: {cls[predicted_class]}")
    print("probability values : ",predicted_class,outputs)
    return outputs