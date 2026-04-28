import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

class PneumoniaModel:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=None) # Avoid pretrained weights when loading our own
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        # Load the weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        except FileNotFoundError:
            print("Warning: Model file not found. Using randomly initialized weights for testing.")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.classes = ["Normal", "Pneumonia"]

    def predict(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # OOD Check: Chest X-rays are grayscale. In RGB format, R=G=B.
        # We can detect generic color images by checking color variance.
        import numpy as np
        img_np = np.array(image)
        # Calculate std dev across the color channel (axis 2)
        channel_std = np.std(img_np, axis=2).mean()
        
        # If the average difference between R,G,B channels is > 5.0 (allowing for slight JPEG artifacts)
        # then it is likely a colored photo, not an X-ray.
        if channel_std > 5.0:
            raise ValueError("The uploaded image does not appear to be a chest X-ray. Please upload a valid grayscale X-ray scan.")
            
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        class_idx = predicted.item()
        return {
            "prediction": self.classes[class_idx],
            "confidence": float(confidence.item()),
            "probabilities": {
                "Normal": float(probabilities[0][0].item()),
                "Pneumonia": float(probabilities[0][1].item())
            }
        }
