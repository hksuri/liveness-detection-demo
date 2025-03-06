"""
evaluate.py

Evaluate the trained model on a test set to compute metrics like accuracy, F1-score, etc.
"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score, f1_score, classification_report

from model import get_model

def evaluate(test_dir, model_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    all_labels = []
    all_preds  = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro')  # or 'binary'
    
    print("Accuracy:", acc)
    print("F1-Score:", f1)
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

if __name__ == "__main__":
    test_dir   = "path/to/preprocessed_data/test"
    model_path = "liveness_model.pth"
    evaluate(test_dir, model_path)