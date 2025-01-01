import torch
import torch.nn as nn
import torchxrayvision as xrv
from torchvision import transforms

class PolicySubsetSelection(nn.Module):
    def __init__(self, output_size=1):
        super(PolicySubsetSelection, self).__init__()
        
        # Load the pre-trained ResNet model
        self.resnet = xrv.models.get_model("resnet50-res512-all")
        self.resnet.model.fc = nn.Identity()

        # Remove the final fully connected layers from ResNet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Optionally, fine-tune ResNet layers (set requires_grad=False)
        for param in self.resnet.parameters():
          param.requires_grad = False
        
        # Additional fully connected layers
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(33, 1)
        
        # Sigmoid layer for binary classification output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, counter=0):
        # Extract features from the pre-trained ResNet
        x = self.resnet(x)
        
        # Flatten the output of ResNet50
        x = x.view(x.size(0), -1)  # (batch_size, feature_size)
        
        # Pass the flattened features through the additional fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        batch_size = x.size(0)
        counter_tensor = torch.full((batch_size, 1), counter, device=x.device)
        x = torch.cat((x, counter_tensor), dim=1)
        x = self.fc4(x)
        
        # Apply sigmoid at the end for binary classification
        x = self.sigmoid(x)
        
        return x
    def update_counters(self, counters, actions):
        """
        Update the counters based on the actions.
        
        Args:
            counters (torch.Tensor): Tensor of size (batch_size, 1) representing the current counters.
            actions (torch.Tensor): Tensor of size (batch_size, 1) representing the actions (0 or 1).
        
        Returns:
            torch.Tensor: Updated counters.
        """
        # Ensure counters and actions are on the same device
        actions = actions.to(counters.device)
        
        # Increment counters where actions are 1
        updated_counters = counters + actions
        
        return updated_counters
