# Medicinal Plant Classification - CNN + ViT (Appendix Code)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from datasets import load_dataset
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# Load dataset
dataset = load_dataset("funkepal/medicinal_plant_images")

# Feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean,
                         std=feature_extractor.image_std)
])

# Dataset class
class PlantDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        label = item['label']
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

train_dataset = PlantDataset(dataset['train'], transform)
test_dataset = PlantDataset(dataset['test'], transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# CNN Feature Extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv_layer(x)

cnn_extractor = CNNFeatureExtractor()

# Hybrid Model
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = cnn_extractor
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=41
        )
        self.conv_proj = nn.Conv2d(128, 3, kernel_size=1)

    def forward(self, x):
        x = self.cnn(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.conv_proj(x)
        outputs = self.vit(pixel_values=x)
        return outputs.logits

model = HybridModel()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

scaler = GradScaler()

# Training function
def train(model, loader):
    model.train()
    total_loss, correct = 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

# Evaluation
def evaluate(model, loader):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

# Training loop
for epoch in range(3):
    train_loss, train_acc = train(model, train_loader)
    test_loss, test_acc = evaluate(model, test_loader)

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
