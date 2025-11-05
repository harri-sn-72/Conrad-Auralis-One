import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTModel
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load datasets
train_dataset = datasets.ImageFolder('../data/fer2013/train', transform=transform)
val_dataset = datasets.ImageFolder('../data/fer2013/validation', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Define the custom ViT classifier
class ViTClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        logits = self.classifier(pooled_output)
        return logits

# Instantiate model
model = ViTClassifier(num_classes=7).to(device)

# Optimizer
optimizer = Adam(model.parameters(), lr=3e-5)

# Tracking best validation accuracy
best_acc = 0.0

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1} Validation Accuracy: {acc:.2f}%")

    # save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"s savedbest model with accuracy: {acc:.2f}%")

print("Training complete. Best accuracy:", best_acc)
