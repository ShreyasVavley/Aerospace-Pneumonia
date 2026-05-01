import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
import cv2
import base64

class PneumoniaModel:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        except FileNotFoundError:
            print("Warning: Model file not found.")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.classes = ["Normal", "Pneumonia"]

    def generate_gradcam(self, input_tensor, target_class):
        # Grad-CAM implementation
        # Get activations from the last convolutional layer
        feature_blobs = []
        def hook_feature(module, input, output):
            feature_blobs.append(output.data.cpu().numpy())
        
        # In ResNet18, layer4 is the last convolutional block
        handle = self.model.layer4.register_forward_hook(hook_feature)
        
        # Forward pass
        input_tensor.requires_grad = True
        output = self.model(input_tensor)
        
        # Backward pass for the target class
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Get gradients
        grads = input_tensor.grad.data.cpu().numpy()
        handle.remove()
        
        # Process features and gradients
        features = feature_blobs[0][0] # (512, 7, 7)
        weights = np.mean(grads[0], axis=(1, 2)) # Global Average Pooling of gradients
        
        cam = np.zeros(features.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * features[i, :, :]
            
        cam = np.maximum(cam, 0) # ReLU
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

    def predict(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Advanced OOD (Out-of-Distribution) Check
        img_np = np.array(image)
        
        # 1. Color Check: Chest X-rays are strictly grayscale. 
        # Standard deviation across the R,G,B channels should be near zero.
        channel_std = np.std(img_np, axis=2).mean()
        
        # 2. Brightness Check: Chest X-rays have a balanced but relatively dark mean intensity.
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        mean_intensity = np.mean(gray_img)
        
        # 3. Contrast Check: Chest X-rays have high structural contrast (ribs/organs vs air).
        contrast_std = np.std(gray_img)

        # Validation Logic:
        # - channel_std > 5.0 -> Too much color (e.g. landscape photo, selfie)
        # - mean_intensity < 20 or > 230 -> Image is mostly black or mostly white (noise/garbage)
        # - contrast_std < 15 -> Image is too flat/blurry (not a high-contrast medical scan)
        
        is_valid = True
        error_msg = ""
        
        if channel_std > 8.0:
            is_valid = False
            error_msg = "Image is in color. Please upload a grayscale medical X-ray scan."
        elif mean_intensity < 15 or mean_intensity > 240:
            is_valid = False
            error_msg = "Image exposure is invalid. Please upload a clear chest X-ray."
        elif contrast_std < 15:
            is_valid = False
            error_msg = "Image has insufficient structural detail. Please upload a valid chest X-ray scan."
            
        if not is_valid:
            raise ValueError(error_msg)
            
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # We need gradients for Grad-CAM
        self.model.eval()
        outputs = self.model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        class_idx = predicted.item()
        
        # Generate Heatmap
        heatmap = self.generate_gradcam(tensor, class_idx)
        
        # Process heatmap for display
        heatmap_img = np.uint8(255 * heatmap)
        heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        
        # Prepare original image for overlay
        orig_img = np.array(image.resize((224, 224)))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        
        # Superimpose
        superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap_img, 0.4, 0)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', superimposed_img)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "prediction": self.classes[class_idx],
            "confidence": float(confidence.item()),
            "heatmap": heatmap_base64,
            "probabilities": {
                "Normal": float(probabilities[0][0].item()),
                "Pneumonia": float(probabilities[0][1].item())
            }
        }
