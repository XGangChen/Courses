import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("GPU Name:", torch.cuda.get_device_name(0))

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Load dataset from local path (you should download and extract the dataset beforehand)
dataset_path = "C:\XGang\Graduation\First_Year\Machine-Learning\WEEK6\Sport-Classification-Dataset" # 20 kinds of sports
train_dataset = ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = ImageFolder(root=f"{dataset_path}/test", transform=transform)

# Use DataLoader to load images
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load all images and labels into memory and move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_features(loader):
    X = []
    y = []
    for images, labels in tqdm(loader):
        images = images.to(device)
        # Flatten images
        features = images.view(images.size(0), -1)
        X.append(features.cpu().numpy())
        y.append(labels.numpy())
    return np.vstack(X), np.concatenate(y)

X_train, y_train = extract_features(train_loader)
X_test, y_test = extract_features(test_loader)

# Encode labels if not numeric
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Train logistic regression (using CPU due to sklearn limitation)
clf = LogisticRegression(max_iter=5000, solver='saga', multi_class='multinomial', verbose=1)
clf.fit(X_train, y_train_encoded)

# Predict and evaluate
y_pred = clf.predict(X_test)
report = classification_report(y_test_encoded, y_pred, target_names=train_dataset.classes, output_dict=False)
report
