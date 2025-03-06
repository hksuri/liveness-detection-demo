"""
train.py

Training loop for the liveness detection CNN.
"""

import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model import get_model

def main(train_dir, val_dir, batch_size=16, num_epochs=10, learning_rate=0.001):
    # Define transforms: e.g., resize, convert to tensor, normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # You can add more (mean/std) normalization if desired
    ])
    
    # Create dataset from folders
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset   = datasets.ImageFolder(val_dir, transform=transform)
    
    # Create data loaders
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = get_model(num_classes=2)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Optionally, if you have a GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct    = 0
        total      = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100.0 * correct / total
        avg_loss  = total_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_correct = 0
        val_total   = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_images)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    # Save the model
    torch.save(model.state_dict(), "liveness_model.pth")
    print("Model saved to liveness_model.pth.")

if __name__ == "__main__":
    train_dir = "path/to/preprocessed_data/train"
    val_dir   = "path/to/preprocessed_data/val"
    main(train_dir, val_dir)