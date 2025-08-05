
# CNN_Classification_with_Dropout.py
# 完整作答：已加入 Dropout 並進行準確率比較

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_set, test_set = random_split(dataset, [48000, 12000])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# CNN Model with Dropout
class SimpleCNN(nn.Module):
    def __init__(self, use_dropout=False):
        super(SimpleCNN, self).__init__()
        self.use_dropout = use_dropout
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        return x

def train(model, loader, optimizer, criterion):
    model.train()
    correct, total, loss_sum = 0, 0, 0
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total, loss_sum / total

def test(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
    return correct / len(loader.dataset)

# 比較兩種模型：無 Dropout vs 有 Dropout
models = {"No Dropout": SimpleCNN(use_dropout=False),
          "With Dropout": SimpleCNN(use_dropout=True)}

results = {}
for label, model in models.items():
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_acc_list, test_acc_list = [], []

    for epoch in range(10):
        train_acc, _ = train(model, train_loader, optimizer, criterion)
        test_acc = test(model, test_loader)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"[{label}] Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

    results[label] = {"train": train_acc_list, "test": test_acc_list}

# 繪圖比較
for label in results:
    plt.plot(results[label]["test"], label=f"{label} Test")
plt.title("Test Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
