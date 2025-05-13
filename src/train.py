import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import BrainTumorCNN, get_resnet18_model
from utils import get_dataloaders
from eval import evaluate_model


def train_model(model, train_loader, test_loader, device, epochs=5, lr=0.001):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Optional: smooth labels
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_test_acc = 0.0  # Track best test accuracy

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # ✅ Train accuracy
        model.eval()
        correct_train, total_train = 0, 0
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train

        # ✅ Test accuracy
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc = 100 * correct_test / total_test

        # ✅ Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")

        print(
            f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, "
            f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%"
        )

    print(f"✅ Training complete. Best Test Accuracy: {best_test_acc:.2f}%")
    return model


# --------------------------
# ✅ Entry point of script
# --------------------------
if __name__ == "__main__":
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, test_loader, class_names = get_dataloaders(
        "data/brain_tumor_dataset", batch_size=16, model_type="resnet"
    )
    print("✅ Data loaded")

    # --------------------------
    # ✅ Switch between models
    # --------------------------
    USE_RESNET = True  # 👈 Change this to False to use BrainTumorCNN

    if USE_RESNET:
        model = get_resnet18_model(num_classes=4)
        model_name = "resnet18_model.pth"
        print("✅ ResNet18 model initialized")
    else:
        model = BrainTumorCNN(num_classes=4)
        model_name = "cnn_model.pth"
        print("✅ Custom CNN model initialized")

    # Train
    trained_model = train_model(
        model, train_loader, test_loader, device, epochs=5, lr=0.001
    )

    # Save the model
    torch.save(trained_model.state_dict(), model_name)
    print(f"💾 Model saved as {model_name}")

    # Evaluate the model on test data
    evaluate_model(trained_model, test_loader, device)
