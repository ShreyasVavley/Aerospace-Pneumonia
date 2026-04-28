import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import kagglehub

def train_model(num_epochs=3, batch_size=32, learning_rate=0.001, max_batches=None):
    """
    Train a ResNet18 model for Pneumonia Detection using Transfer Learning.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    print("Downloading/Locating dataset via kagglehub...")
    try:
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print(f"Dataset located at: {path}")
        # The dataset typically extracts into a 'chest_xray' or 'chest_xray/chest_xray' subfolder
        base_dir = path
        if os.path.exists(os.path.join(path, "chest_xray")):
            base_dir = os.path.join(path, "chest_xray")
        
        train_dir = os.path.join(base_dir, 'train')
        val_dir = os.path.join(base_dir, 'val')
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        train_dir, val_dir = None, None

    # Data Augmentation and Normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    try:
        if not train_dir or not os.path.exists(train_dir):
            raise Exception("Train directory not found.")
            
        image_datasets = {
            'train': datasets.ImageFolder(train_dir, data_transforms['train']),
            'val': datasets.ImageFolder(val_dir, data_transforms['val'])
        }
        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
            'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0)
        }
    except Exception as e:
        print("Dataset not found. Please ensure the dataset is in the correct path.")
        print("Skipping actual training loop, saving an initialized model for API testing...")
        # Save a dummy model so the API has something to load
        model = models.resnet18(weights='DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        torch.save(model.state_dict(), 'pneumonia_model.pth')
        print("Dummy model saved as pneumonia_model.pth")
        return

    # Load Pretrained ResNet18
    model = models.resnet18(weights='DEFAULT')
    
    # Freeze early layers if necessary (optional, but good for transfer learning)
    # for param in model.parameters():
    #     param.requires_grad = False

    # Replace the final fully connected layer (2 classes: Normal vs Pneumonia)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                if max_batches and batch_idx >= max_batches:
                    break

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print("Training complete.")
    torch.save(model.state_dict(), 'pneumonia_model.pth')
    print("Model saved as pneumonia_model.pth")

if __name__ == '__main__':
    train_model()
