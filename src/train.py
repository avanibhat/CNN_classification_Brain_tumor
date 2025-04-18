import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import BrainTumorCNN
from utils import get_dataloaders
from eval import evaluate_model


def train_model(model, train_loader, test_loader, device, epochs=5, lr=0.001):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward + Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate train accuracy after each epoch
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(
            f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%"
        )

    print("✅ Training complete.")
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
        "data/brain_tumor_dataset", batch_size=16
    )
    print("✅ Data loaded")

    # Initialize model
    model = BrainTumorCNN(num_classes=4)
    print("✅ Model initialized")

    # Train
    trained_model = train_model(
        model, train_loader, test_loader, device, epochs=10, lr=0.001
    )

    # Save the model
    torch.save(trained_model.state_dict(), "cnn_model.pth")
    print("💾 Model saved as cnn_model.pth")

    # Evaluate the model on test data
    evaluate_model(trained_model, test_loader, device)
