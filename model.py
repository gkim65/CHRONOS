import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
# import cv2
import os
import requests
from io import BytesIO
from PIL import Image

# Function to fetch satellite images from a free dataset (Sentinel-2 via OpenEarth API)
def fetch_satellite_image(lat, lon, zoom=12):
    url = f"https://api.openearth.community/sentinel2?lat={lat}&lon={lon}&zoom={zoom}"
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        return img
    else:
        print("Failed to fetch satellite image")
        return None

# U-Net model for change detection
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder_layers = list(self.encoder.children())[:-2]  # Remove FC layers
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.upsample = nn.ConvTranspose2d(512, 1, kernel_size=1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.upsample(x)
        return torch.sigmoid(x)

# Load dataset from OpenEarth (Placeholder coordinates)
class SatelliteDataset(Dataset):
    def __init__(self, locations, transform=None):
        self.locations = locations
        self.transform = transform
    
    def __len__(self):
        return len(self.locations)
    
    def __getitem__(self, idx):
        lat, lon = self.locations[idx]
        img = fetch_satellite_image(lat, lon)
        if img is not None:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        else:
            return None

# Training pipeline (Using OpenEarth dataset)
def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])
    dataset = SatelliteDataset([(37.7749, -122.4194), (34.0522, -118.2437)], transform=transform)  # SF & LA
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = UNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        for images in dataloader:
            if images is None:
                continue  # Skip if API fails
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)  # Placeholder loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

if __name__ == "__main__":
    train()
