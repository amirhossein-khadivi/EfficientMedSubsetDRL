import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchxrayvision as xrv

#create our data structure
class CustomDataset(Dataset):
    def __init__(self, data_json, transform=None):
        self.data = data_json
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['image_path']
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.data[idx]['label_vector'], dtype=torch.float32)
        counter = torch.tensor(self.data[idx]['counter'], dtype=torch.float32)

        return image, label, counter


#AlexNet as the agent
class AlexNetGrayscale(nn.Module):
    def __init__(self, num_classes=1):
        super(AlexNetGrayscale, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            #nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.sigmoid(x)

        return x



#ResNet as the predictor
class CustomClassifier(nn.Module):
    def __init__(self, output_size=14):
        super(CustomClassifier, self).__init__()
        self.resnet = xrv.models.get_model("resnet50-res512-all")
        self.resnet.model.fc = nn.Identity()

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        for param in self.resnet.parameters():
            param.requires_grad = False

        resnet_output_size = 1 * 512 * 512
        self.fc1 = nn.Linear(resnet_output_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        #print("ResNet output shape:", x.shape)
        x = x.view(x.size(0), -1)
        #print("After flattening:", x.shape)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        x = self.sigmoid(x)

        return x



#----------------preparing for train process------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with open('/content/Train_dataset/dataset.json', 'r') as f:
    dataset_json = json.load(f)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0], [1])
])

transform_model = transforms.Compose([
    transforms.Resize((512, 512)),
])

dataset = CustomDataset(dataset_json, transform=transform)

#ctrate Main Batches
main_batches = torch.utils.data.random_split(
    dataset,
    [len(dataset) // 80 + (1 if i < len(dataset) % 80 else 0) for i in range(80)]
)


#call Agent, Model, optimizer
agent = AlexNetGrayscale().to(device)
model = CustomClassifier().to(device)
model_B = CustomClassifier().to(device)

agent_optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model_B_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCELoss()
criterion_B = torch.nn.BCELoss()






#-----------------Training pipline----------------
import matplotlib.pyplot as plt
episode_rewards = []
validation_losses = []
selection_rates = []

# Training pipeline
for episode in range(100):
    episode_reward = 0
    episode_selection_rate = []

    for step in range(80):
        #step1:
        train_batch = DataLoader(main_batches[step], batch_size=len(main_batches[step]), shuffle=True)
        val_batch = DataLoader(main_batches[(step + 1) % 80], batch_size=len(main_batches[(step + 1) % 80]), shuffle=False)

        #step2:
        #agent.eval()
        selected_probs = []
        counters = []
        actions_list = []

        for images, labels, _ in train_batch:
            images, labels = images.to(device), labels.to(device)

            probs = agent(images)

            actions = (probs > 0.5).float()
            '''
            if step < 10:
              actions = (probs > (probs.sum()/len(probs)).item()).float()
            else:
              actions = (probs > 0.5).float()

            '''

            selected_probs.append(probs)
            actions_list.append(actions)
            #counters.append(counter)

        selected_probs = torch.cat(selected_probs)
        actions_list = torch.cat(actions_list)
        #counters = torch.cat(counters)
        '''
        selected_data = [
            (images[i], labels[i], counters[i])
            for i in range(len(images))
            if (actions[i] > 0.5).item()
        ]
        '''
        selected_data = []
        for i in range(len(images)):
          if (actions[i] == 1).item():
            selected_data.append((images[i],labels[i]))


        selection_rate = actions_list.mean().item() * 100
        episode_selection_rate.append(selection_rate)

        if len(selected_data) == 0:
            print((probs > 0.5).sum().item())
            print("NO subset")
            continue

        train_subset = DataLoader(selected_data, batch_size=32, shuffle=True)

        #Step 3:
        model.train()
        if step % 8 ==0:
          model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

        epoch_losses = []
        epoch_loss = 0.0
        for images, labels in train_subset:
            images, labels = images.to(device), labels.to(device)

            outputs = model(transform_model(images))
            loss = criterion(outputs, labels)

            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            epoch_loss += loss.item()

        epoch_losses.append(epoch_loss / len(train_subset))
        print(f"Episode {episode + 1}, Step {step + 1}")

        #Step 3.1:
        train_batch_B = DataLoader(main_batches[step], batch_size= 128, shuffle=True)
        model_B.train()
        if step % 8 == 0:
          model_B.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)


        epoch_loss_B = 0.0
        for images, labels, _ in train_batch_B:
            images, labels = images.to(device), labels.to(device)
            outputs_B = model(transform_model(images))
            loss = criterion_B(outputs_B, labels)

            model_B_optimizer.zero_grad()
            loss.backward()
            model_B_optimizer.step()

            epoch_loss_B += loss.item()

        #add loss-batch

        #step4:
        model.eval()
        val_loss = 0.0
        loss_B = 0.0
        with torch.no_grad():
            for images, labels, _ in val_batch:
                images, labels = images.to(device), labels.to(device)
                outputs = model(transform_model(images))
                val_loss += criterion(outputs, labels).item()

            for images, labels, _ in val_batch:
                images_B, labels_B = images.to(device), labels.to(device)
                outputs_B = model_B(transform_model(images_B))
                loss_B += criterion_B(outputs_B, labels_B).item()

        loss_B /= len(train_batch)
        val_loss /= len(val_batch)
        validation_losses.append(val_loss)

        #reward = abs(val_loss - loss_B) + (actions.sum().item() / len(train_batch))
        reward = abs(val_loss - loss_B) + (len(selected_data)/1089)
        episode_reward += reward

        #step5:
        agent.train()
        agent_optimizer.zero_grad()

        reward_tensor = torch.tensor(reward, device=selected_probs.device, dtype=torch.float32)
        #reward_tensor = -torch.tensor(reward, device=selected_probs.device, dtype=torch.float32)
        #agent_loss = -torch.mean(reward_tensor * torch.log(selected_probs + 1e-8))
        #agent_loss = -torch.mean(reward_tensor)
        reward_tensor.requires_grad_()
        agent_loss = reward_tensor
        agent_loss.backward()
        agent_optimizer.step()

        print(f"Episode {episode + 1}, Step {step + 1}: Val Loss = {val_loss:.4f}, Reward = {reward:.4f}, agent_loss = {agent_loss:.4f}, ABS_loss = {abs(val_loss - loss_B)},Selection Rate = {len(selected_data):.2f}%")

    episode_rewards.append(episode_reward)
    selection_rates.append(np.mean(episode_selection_rate))

    print(f"Episode {episode + 1} Complete: Total Reward = {episode_reward:.4f}, Avg Selection Rate = {np.mean(episode_selection_rate):.2f}%")


'''
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Training pipeline
for episode in range(100):
    episode_reward = 0
    episode_selection_rate = []

    for step in range(80):
        #step1:
        train_batch = DataLoader(main_batches[step], batch_size=len(main_batches[step]), shuffle=True)
        val_batch = DataLoader(main_batches[(step + 1) % 80], batch_size=len(main_batches[(step + 1) % 80]), shuffle=False)

        #step2:
        #agent.eval()
        selected_probs = []
        counters = []
        actions_list = []

        for images, labels, _ in train_batch:
            images, labels = images.to(device), labels.to(device)

            probs = agent(images)
            #init_threshold = 0.5
            #final_threshold = 0.7
            #threshold = init_threshold + (final_threshold - init_threshold) * ((episode+step) / 180)
            actions = (probs > 0.5).float()
            '''
            if step < 10:
              actions = (probs > (probs.sum()/len(probs)).item()).float()
            else:
              actions = (probs > 0.5).float()

            '''

            selected_probs.append(probs)
            actions_list.append(actions)
            #counters.append(counter)

        selected_probs = torch.cat(selected_probs)
        actions_list = torch.cat(actions_list)
        #counters = torch.cat(counters)
        '''
        selected_data = [
            (images[i], labels[i], counters[i])
            for i in range(len(images))
            if (actions[i] > 0.5).item()
        ]
        '''
        selected_data = []
        for i in range(len(images)):
          if (actions[i] == 1).item():
            selected_data.append((images[i],labels[i]))


        selection_rate = actions_list.mean().item() * 100
        episode_selection_rate.append(selection_rate)

        if len(selected_data) == 0:
            print((probs > 0.5).sum().item())
            print("NO subset")
            continue

        train_subset = DataLoader(selected_data, batch_size=32, shuffle=True)

        #Step 3:
        model.train()
        if step == 20:
          model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

        epoch_losses = []
        epoch_loss = 0.0
        for images, labels in train_subset:
            images, labels = images.to(device), labels.to(device)

            outputs = model(transform_model(images))
            loss = criterion(outputs, labels)

            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            epoch_loss += loss.item()

        epoch_losses.append(epoch_loss / len(train_subset))
        print(f"Episode {episode + 1}, Step {step + 1}")

        #Step 3.1:
        train_batch_B = DataLoader(main_batches[step], batch_size= 128, shuffle=True)
        model_B.train()
        if step == 20:
          model_B.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)


        epoch_loss_B = 0.0
        for images, labels, _ in train_batch_B:
            images, labels = images.to(device), labels.to(device)
            outputs_B = model(transform_model(images))
            loss = criterion_B(outputs_B, labels)

            model_B_optimizer.zero_grad()
            loss.backward()
            model_B_optimizer.step()

            epoch_loss_B += loss.item()

        #add loss-batch

        #step4:
        model.eval()
        val_loss = 0.0
        loss_B = 0.0
        with torch.no_grad():
            for images, labels, _ in val_batch:
                images, labels = images.to(device), labels.to(device)
                outputs = model(transform_model(images))
                val_loss += criterion(outputs, labels).item()

            for images, labels, _ in val_batch:
                images_B, labels_B = images.to(device), labels.to(device)
                outputs_B = model_B(transform_model(images_B))
                loss_B += criterion_B(outputs_B, labels_B).item()

        loss_B /= len(train_batch)
        val_loss /= len(val_batch)
        validation_losses.append(val_loss)

        #reward = abs(val_loss - loss_B) + (actions.sum().item() / len(train_batch))
        #reward = abs(val_loss - loss_B) + 0.5*((len(selected_data)/1089) - 0.8)
        #reward = abs(val_loss - loss_B) + 0.5 * abs((len(selected_data)/len(main_batches[step])) - 0.8)
        #----->reward = (2*(val_loss - loss_B))**2 + 0.4 * ((len(selected_data) / len(main_batches[step])) - 0.8)**2 - 0.2*((len(selected_data) / len(main_batches[step])) - 0.8)**2
        reward = 0.1*abs(val_loss - loss_B) + 0.9*(len(selected_data) / len(main_batches[step]))
        episode_reward += reward

        #step5:
        agent.train()
        agent_optimizer.zero_grad()

        reward_tensor = torch.tensor(reward, device=selected_probs.device, dtype=torch.float32)
        #reward_tensor = -torch.tensor(reward, device=selected_probs.device, dtype=torch.float32)
        agent_loss = -torch.mean(reward_tensor * torch.log(selected_probs))      #agent_loss = -torch.mean(reward_tensor)
        #reward_tensor.requires_grad_()
        #agent_loss = reward_tensor
        agent_loss.backward()
        agent_optimizer.step()

        print(f"Episode {episode + 1}, Step {step + 1}: Val Loss = {val_loss:.4f}, Reward = {reward:.4f}, agent_loss = {agent_loss:.4f}, ABS_loss = {abs(val_loss - loss_B)},Selection Rate = {len(selected_data):.2f}%")

    episode_rewards.append(episode_reward)
    selection_rates.append(np.mean(episode_selection_rate))

    print(f"Episode {episode + 1} Complete: Total Reward = {episode_reward:.4f}, Avg Selection Rate = {np.mean(episode_selection_rate):.2f}%")

'''








