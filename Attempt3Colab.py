# 1. Mount Google Drive (force remount to avoid errors)
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

!unzip -q "/content/drive/MyDrive/Fer2013.zip" -d "/content"

!cp "/content/drive/MyDrive/train_vit_3.py" "/content/train_vit_3.py"

!ls /content/train
!ls /content/test

!python /content/train_vit_3.py

from google.colab import drive
drive.mount('/content/drive')


from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from transformers import ViTModel
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
from IPython.display import clear_output
import matplotlib.pyplot as plt
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

extract_dir = "/content/drive/MyDrive/affectnet"  # Path to unzipped folder
print("Detected classes:", sorted([d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]))

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class AffectNetFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []


        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_file in os.listdir(cls_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, img_file), self.class_to_idx[cls_name]))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# -------------------------
# Load dataset and compute labels
# -------------------------
full_dataset = AffectNetFolderDataset(extract_dir, transform=train_transform)
labels = np.array([lbl for _, lbl in full_dataset])
num_classes = len(full_dataset.classes)
total_samples = len(full_dataset)
print(f"Total samples: {total_samples}, Number of classes: {num_classes}")


# -------------------------
# Class weights (safe)
# -------------------------
label_counts = Counter(labels)
class_weights = [total_samples / max(1, label_counts[i]) for i in range(num_classes)]
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Class weights:", class_weights)


# -------------------------
# Stratified split: train/val/test
# -------------------------
indices = np.arange(total_samples)
train_idx, temp_idx, train_labels, temp_labels = train_test_split(
    indices, labels, test_size=0.2, stratify=labels, random_state=42
)
val_idx, test_idx, val_labels, test_labels = train_test_split(
    temp_idx, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)


train_dataset = Subset(AffectNetFolderDataset(extract_dir, transform=train_transform), train_idx)
val_dataset = Subset(AffectNetFolderDataset(extract_dir, transform=val_test_transform), val_idx)
test_dataset = Subset(AffectNetFolderDataset(extract_dir, transform=val_test_transform), test_idx)


# -------------------------
# Dataloaders
# -------------------------
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, pin_memory=True)


class ViTClassifier(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(ViTClassifier, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)


    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


model = ViTClassifier().to(device)

optimizer = Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss(weight=class_weights)
scaler = torch.cuda.amp.GradScaler()


num_epochs = 30
best_val_acc = 0.0
train_acc_list = []
val_acc_list = []


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0


    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        try:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        except Exception as e:
            print(f"Skipping a batch due to error: {e}")


    avg_loss = total_loss / len(train_loader)
    train_acc = correct_train / total_train
    train_acc_list.append(train_acc)


    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            try:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
            except Exception as e:
                print(f"Skipping a validation batch due to error: {e}")
    val_acc = correct_val / total_val
    val_acc_list.append(val_acc)


    clear_output(wait=True)
    plt.figure(figsize=(8,6))
    plt.plot(range(1, epoch+2), train_acc_list, label="Training Accuracy")
    plt.plot(range(1, epoch+2), val_acc_list, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


    print(f"Epoch {epoch+1} training loss: {avg_loss:.4f}, training accuracy: {train_acc:.4f}")
    print(f"Epoch {epoch+1} validation accuracy: {val_acc:.4f}")


    save_path = f"/content/drive/MyDrive/affectnet_epoch{epoch+1}.pth"
    torch.save(model.state_dict(), save_path)


    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_path = "/content/drive/MyDrive/affectnet_best_model.pth"
        torch.save(model.state_dict(), best_path)
        print(f"Best model updated and saved to {best_path}")


    print(f"Epoch {epoch+1} completed.\n")

