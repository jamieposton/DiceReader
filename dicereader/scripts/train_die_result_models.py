import os
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


BASENAME = os.getcwd()
DATA_ROOT = os.path.join(BASENAME, 'data', 'labeled_data', 'die_result')
CHECKPOINTS_DIR = os.path.join(BASENAME, 'checkpoints', "die_result_models")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
IMG_SIZE = 224  # EfficientNet-B0 default
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def train_die_type_model(die_type, data_dir):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    class_names = dataset.classes
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"{die_type} Epoch {epoch+1}/{NUM_EPOCHS}"):
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

        print(f"{die_type} Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(CHECKPOINTS_DIR, f'effnet_{die_type}_result_{timestamp}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, save_path)
    print(f"Model for {die_type} saved to {save_path}")

if __name__ == "__main__":
    for die_type in sorted(os.listdir(DATA_ROOT)):
        die_type_dir = os.path.join(DATA_ROOT, die_type)
        if not os.path.isdir(die_type_dir):
            continue
        print(f"Training model for die type: {die_type}")
        train_die_type_model(die_type, die_type_dir)
