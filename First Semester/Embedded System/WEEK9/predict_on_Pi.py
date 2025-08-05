from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(x)  # 2nd pooling layer to get 7x7 output
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load model
model = Net()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))
model.eval()

# Load and preprocess image
img = Image.open("mydigit.png").convert("L")  # grayscale
img.show()  # Show the input image

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Predict
with torch.no_grad():
    output = model(img_tensor)
    pred = output.argmax(dim=1)
    print(f"Prediction: {pred.item()}")
