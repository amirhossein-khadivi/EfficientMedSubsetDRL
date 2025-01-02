import torch
import torch.nn as nn
import torchxrayvision as xrv
from torchvision import transforms
import torch.nn.functional as F

class CustomClassifier(nn.Module):
    def __init__(self, output_size=14):
        super(CustomClassifier, self).__init__()

        # Load the pre-trained ResNet model
        self.resnet = xrv.models.get_model("resnet50-res512-all")
        self.resnet.model.fc = nn.Identity()

        # Remove the final fully connected layers from ResNet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Optionally, fine-tune ResNet layers (set requires_grad=False)
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Additional fully connected layers with new architecture
        self.fc1 = nn.Linear(2048, 512)  # 2048 to 512
        self.fc2 = nn.Linear(512, 128)   # 512 to 128
        self.fc3 = nn.Linear(128, output_size)  # Output layer for final classification

        # Sigmoid layer for binary classification output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Extract features from the pre-trained ResNet
        x = self.resnet(x)

        # Flatten the output of ResNet50
        x = x.view(x.size(0), -1)  # (batch_size, feature_size)

        # Pass the flattened features through the additional fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # Apply sigmoid at the end for binary classification
        x = self.sigmoid(x)

        return x