import torch.nn as nn
import torch.nn.functional as F


class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super(BrainTumorCNN, self).__init__()

        # First convolution layer:
        # - Input: 3 color channels (RGB)
        # - Output: 16 feature maps
        # - Kernel size: 3x3
        # - Padding: 1 (to preserve spatial size)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

        # Pooling layer (used after each conv layer to reduce spatial dimensions)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves the height & width

        # Second convolution layer:
        # - Input: 16 feature maps
        # - Output: 32 feature maps
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )

        # Third convolution layer:
        # - Input: 32 feature maps
        # - Output: 64 feature maps
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )

        # Fully connected (dense) layer:
        # Input: flattened feature map of shape 64 x 28 x 28 = 50176
        # Output: 128 hidden units
        self.fc1 = nn.Linear(in_features=64 * 28 * 28, out_features=128)

        # Dropout layer to reduce overfitting during training
        self.dropout = nn.Dropout(p=0.3)

        # Output layer:
        # Input: 128 units from previous FC layer
        # Output: `num_classes` (i.e., 4 in our case — one for each tumor type)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # Apply conv -> ReLU -> pool repeatedly
        x = self.pool(F.relu(self.conv1(x)))  # Output: [B, 16, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # Output: [B, 32, 56, 56]
        x = self.pool(F.relu(self.conv3(x)))  # Output: [B, 64, 28, 28]

        # Flatten the feature maps into a vector for the fully connected layer
        x = x.view(-1, 64 * 28 * 28)

        # Fully connected layer with ReLU
        x = F.relu(self.fc1(x))

        # Apply dropout
        x = self.dropout(x)

        # Final output layer (no activation — softmax will be applied in loss function)
        x = self.fc2(x)

        return x
