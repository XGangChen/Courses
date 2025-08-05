import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Dataset Paths
data_dir = "/home/xgang/XGang/Graduation/First_Year/Reinforcement-Learning/Week11/ANTS"  # Replace with your dataset path

# Transformations for preprocessing images
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Load the dataset using torchvision
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Define the model (DN)
class DN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        pooledw = convw // 2
        pooledh = convh // 2

        linear_input_size = pooledw * pooledh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.head(x))

# Hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 1e-4
num_splits = 5  # For cross-validation

# Cross-validation setup
kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}/{num_splits}")

    # Split dataset into train and validation subsets
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = DN(h=64, w=64, outputs=1)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            labels = labels.float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                labels = labels.float().unsqueeze(1)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save results for this fold
    fold_results.append((train_losses, val_losses))

    # Plot convergence for this fold
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold + 1} Loss Convergence")
    plt.legend()
    plt.show()

# Average results across all folds
avg_train_loss = [sum(fold[0][epoch] for fold in fold_results) / num_splits for epoch in range(num_epochs)]
avg_val_loss = [sum(fold[1][epoch] for fold in fold_results) / num_splits for epoch in range(num_epochs)]

# Plot average loss convergence
plt.figure(figsize=(8, 6))
plt.plot(avg_train_loss, label="Average Train Loss")
plt.plot(avg_val_loss, label="Average Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Cross-Validation Loss Convergence")
plt.legend()
plt.show()
