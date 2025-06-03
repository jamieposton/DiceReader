import os
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Config
import glob
BASENAME = os.getcwd()
from datetime import datetime
# Use all curated label folders
DATA_DIR = os.path.join(BASENAME, 'data', 'curated-labels')
print(f"Using all curated data in: {DATA_DIR}")

class AllRunsImageFolder(datasets.ImageFolder):
    def __init__(self, root, **kwargs):
        # root should be 'data/curated-labels'
        super().__init__(root, **kwargs)

# Use the custom dataset to load all images recursively
CHECKPOINTS_DIR = os.path.join(BASENAME, 'checkpoints')
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
MODEL_SAVE_PATH = os.path.join(CHECKPOINTS_DIR, f'efficientnet_dice_classifier_{timestamp}.pt')
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_CLASSES = 6
IMG_SIZE = 224  # EfficientNet-B0 default
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset = AllRunsImageFolder(DATA_DIR, transform=transform)

class_names = dataset.classes
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Model
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    train_loss = running_loss / total
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)
    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names
}, MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
