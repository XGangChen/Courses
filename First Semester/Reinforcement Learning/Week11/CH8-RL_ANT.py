import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define data transformations (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 64x64
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values to [-1, 1]
])

# Paths to dataset
data_dir = "/home/xgang/XGang/Graduation/First_Year/Reinforcement-Learning/Week11/ANTS"
train_dir = os.path.join(data_dir, "bbox_train")
test_dir = os.path.join(data_dir, "bbox_test")

# Load datasets
train_dataset = ImageFolder(train_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)

# Create DatakLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  

# Update the modal for image input dimensions
class DN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)   # 3 channels for RGB
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        # self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        # self.bn4 = nn.BatchNorm2d(32)
        # self.conv5 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=0)
        # self.bn5 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)

        # Compute the size of the feature map after the convolutional layers
        def conv2d_size_out(size, kernel_size=5, stride=2, padding=2):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        pooledw = convw // 2  # Pooling reduces dimensions by 2
        pooledh = convh // 2

        linear_input_size = pooledw * pooledh * 32  # 32 is the number of filters in the last Conv layer
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.conv4(x)))
        # x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1) 

        return torch.sigmoid(self.head(x))

def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

# -------------------------------------------------------------------------------------
# We need to initialize the model, loss function, and optimizer
DN_model = DN(h=128, w=128, outputs=1)   # Input image size is 64x64
initialize_weights(DN_model)

# Loss and optimizer
loss_function = nn.MSELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.SGD(DN_model.parameters(), lr=0.1, momentum=0.9)
# optimizer = torch.optim.Adam(DN_model.parameters(), lr=0.001, momentum=0.01)
# optimizer = torch.optim.Adagrad(DN_model.parameters(), lr=0.001)
# optimizer = torch.optim.RMSprop(DN_model.parameters(), lr=0.001, momentum=0.01)
optimizer.step()
# The main loop for training our model
max_epochs = 100

loss_values = []

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


for epoch in range(max_epochs):
    DN_model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        labels = labels.float().unsqueeze(1)    # Reshape labels for BCELoss
        optimizer.zero_grad()   # Initialize the gradients as zero
        outputs = DN_model(inputs)  # Forward Propagation
        loss = loss_function(outputs, labels)  # Calculate the loss
        loss.backward()  # Backpropagation (compute gradients)
        optimizer.step()  # Update the weights based on gradients
        running_loss += loss.item()

    scheduler.step()
    epoch_loss = running_loss / len(train_loader)
    loss_values.append(epoch_loss)

    print(f"Epoch => {epoch + 1}/{max_epochs}, Loss: {epoch_loss:.4f}")

# # Plot the loss convergence
# plt.figure(figsize=(8, 6))
# plt.plot(loss_values, label='Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss value for Slides Example')
# plt.legend()
# plt.show()

def smooth_curve(values, weight=0.95):
    smoothed = []
    last = values[0]
    for value in values:
        smoothed_value = last * weight + (1 - weight) * value
        smoothed.append(smoothed_value)
        last = smoothed_value
    return smoothed

smoothed_loss = smooth_curve(loss_values)
plt.plot(loss_values, label="Original Loss")
plt.plot(smoothed_loss, label="Smoothed Loss", linestyle="--")
plt.legend()
plt.show()

# Test the model
DN_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        labels = labels.unsqueeze(1).float()
        outputs = DN_model(inputs)
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")