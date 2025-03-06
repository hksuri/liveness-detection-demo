"""
inference.py

Script to run real-time liveness detection using your webcam.
"""

import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

from model import get_model

def main(model_path="liveness_model.pth"):
    # Load model
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Define transformation for each frame
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # OpenCV face detector (Haar cascade or any other)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for face detection
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face region
            face_rgb = frame[y:y+h, x:x+w]
            
            # Transform face for model
            face_tensor = transform(face_rgb)
            face_tensor = face_tensor.unsqueeze(0)  # shape: (1, 3, 224, 224)
            
            # Inference
            with torch.no_grad():
                output = model(face_tensor)
                _, predicted = torch.max(output.data, 1)
                label = predicted.item()
            
            # Label interpretation: 0 -> real, 1 -> spoof (depends on how dataset is labeled)
            # Check your dataset's class indices in the train_dataset.classes
            # For example, if "real" is index 0 and "spoof" is index 1:
            if label == 0:
                text = "Live"
            else:
                text = "Spoof"
            
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow("Liveness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()