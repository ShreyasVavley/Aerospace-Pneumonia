import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import kagglehub
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_model(num_epochs=10, batch_size=32, learning_rate=0.001, max_batches=None):
    """
    Train a ResNet18 model for Pneumonia Detection using Transfer Learning (Feature Extraction).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    
    logger.info("Downloading/Locating dataset via kagglehub...")
    try:
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        logger.info(f"Dataset located at: {path}")
        base_dir = path
        if os.path.exists(os.path.join(path, "chest_xray")):
            base_dir = os.path.join(path, "chest_xray")
        
        train_dir = os.path.join(base_dir, 'train')
        val_dir = os.path.join(base_dir, 'val')
        test_dir = os.path.join(base_dir, 'test')
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        train_dir, val_dir = None, None

    # Data Augmentation and Normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
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
            
        # Load the full training folder (which contains the bulk of the data)
        full_train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
        
        # Calculate split lengths (90% train, 10% val)
        total_size = len(full_train_dataset)
        train_size = int(0.9 * total_size)
        val_size = total_size - train_size
        
        # Generate split indices
        indices = list(range(total_size))
        import random
        random.seed(42)
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create Subsets
        val_base_dataset = datasets.ImageFolder(train_dir, data_transforms['val'])
        
        image_datasets = {
            'train': torch.utils.data.Subset(full_train_dataset, train_indices),
            'val': torch.utils.data.Subset(val_base_dataset, val_indices),
            'test': datasets.ImageFolder(test_dir, data_transforms['val'])
        }
        
        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2),
            'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=2),
            'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=2)
        }
        logger.info(f"Full Training: Using all images for {num_epochs} epochs.")
    except Exception as e:
        logger.error(f"Dataset division failed: {e}")
        return

    # Load Pretrained ResNet18
    model = models.resnet18(weights='DEFAULT')
    
    # FREEZE BACKBONE: Only train the final layer for maximum speed
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer (2 classes: Normal vs Pneumonia)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Only optimize the parameters of the final layer
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0

    # Training Loop
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch}/{num_epochs - 1}')
        logger.info('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_samples = 0

            # Limit training batches to 50 for speed
            current_max_batches = None

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                if current_max_batches and batch_idx >= current_max_batches:
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
                running_samples += inputs.size(0)
                
                # Progress logging
                if batch_idx % 10 == 0:
                    logger.info(f'Epoch {epoch} [{phase}] Batch {batch_idx}/{len(dataloaders[phase])} - Loss: {loss.item():.4f}')

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects.double() / running_samples

            logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'pneumonia_model.pth'))
                logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

    logger.info("Training complete. Performing final evaluation on test set...")
    
    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'pneumonia_model.pth')))
    model.eval()
    
    test_loss = 0.0
    test_corrects = 0
    
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
        test_loss += loss.item() * inputs.size(0)
        test_corrects += torch.sum(preds == labels.data)
        
    final_test_loss = test_loss / len(image_datasets['test'])
    final_test_acc = test_corrects.double() / len(image_datasets['test'])
    
    logger.info(f'Final Test Loss: {final_test_loss:.4f} Acc: {final_test_acc:.4f}')
    logger.info("Ready for production.")

if __name__ == '__main__':
    train_model()

