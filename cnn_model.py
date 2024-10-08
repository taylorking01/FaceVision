import torch.nn as nn
import torch.nn.functional as F

class FaceDetectionCNN(nn.Module):
    def __init__(self):
        super(FaceDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input channels: 1 for grayscale
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjust based on image dimensions
        self.fc2 = nn.Linear(128, 2)  # Output layer for 2 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 16, 16]
        x = x.view(-1, 64 * 16 * 16)          # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
