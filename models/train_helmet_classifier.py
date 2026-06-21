"""
Train Helmet Classifier — TrafficSentinel AI.

Trains a lightweight custom MobileNetV3-Small binary classifier on the cropped
rider head dataset (helmet vs no_helmet) to achieve high recall and precision.
"""

import os
import sys
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_model(epochs=15, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Dataset paths
    dataset_dir = PROJECT_ROOT / "data" / "classifier_dataset"
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        logger.error(f"Classifier dataset not found. Run utils/prepare_classifier_dataset.py first.")
        return

    # Transforms (data augmentation for training, resize + normalise for both)
    # Rider crops can be small, so 128x128 is a great size (retains detail, fast to train)
    IMG_SIZE = 128
    
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets & Dataloaders
    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    logger.info(f"Class names: {class_names} (should be ['helmet', 'no_helmet'])")
    logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Load SOTA lightweight MobileNetV3 model
    # MobileNetV3-Small is optimized for real-time edge devices (CCTV cameras, mobile nodes)
    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        logger.info("Loaded pretrained MobileNetV3-Small weights.")
    except Exception as e:
        logger.warning(f"Could not download pretrained weights: {e}. Training from scratch.")
        model = models.mobilenet_v3_small(weights=None)

    # Modify the classifier head for 2 classes (helmet vs no_helmet)
    # The output features of mobilenet_v3_small classifier's final layer is 1000
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(class_names))
    model = model.to(device)

    # Loss function and optimizer
    # We use weighted loss if classes are unbalanced (though they are 600 vs 600, so balanced)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    weights_dir = PROJECT_ROOT / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = weights_dir / "custom_helmet_classifier.pth"

    logger.info("Starting training loop...")
    for epoch in range(epochs):
        t0 = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)

        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_total += inputs.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_corrects.double() / val_total

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', pos_label=class_names.index('no_helmet')
        )

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2%} | "
            f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.2%} F1: {f1:.2%}"
        )

        # Save weights if validation accuracy improved
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), str(best_model_path))
            logger.info(f"Saved new best model with Validation Acc = {best_acc:.2%}")

    logger.info(f"Training complete! Best Validation Accuracy: {best_acc:.2%}")
    logger.info(f"Model saved to {best_model_path}")


if __name__ == "__main__":
    train_model(epochs=15, batch_size=32, lr=1e-4)
