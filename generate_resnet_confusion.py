import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import sys
import os

sys.path.append(os.path.abspath("src"))

from utils import get_dataloaders
from model import get_resnet18_model

# Load class names
_, test_loader, class_names = get_dataloaders("data/brain_tumor_dataset", batch_size=16, model_type="resnet")

# Load trained model
model = get_resnet18_model(num_classes=len(class_names))
model.load_state_dict(torch.load("resnet18_model.pth", map_location="cpu"))
model.eval()

# Get predictions and true labels
y_true = []
y_pred = []

print("Generating predictions...")
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

# Plot it
plt.figure(figsize=(12, 10))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - ResNet18 Brain Tumor Classifier (10 Epochs)", fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/resnet18_confusion_matrix_10epochs.png", dpi=300, bbox_inches='tight')
print("Confusion matrix saved to outputs/resnet18_confusion_matrix_10epochs.png")
plt.close()

# Print classification report
print("\n" + "="*60)
print("Classification Report - ResNet18 (10 Epochs)")
print("="*60)
print(classification_report(y_true, y_pred, target_names=class_names))
