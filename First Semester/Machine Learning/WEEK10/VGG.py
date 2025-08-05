import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import os

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the training data
train_data = datasets.ImageFolder(root='/media/xgang/XGang-1T/Graduation/First_Year/Machine-Learning/WEEK10/flower_dataset/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

#-------------Known Networks-------------
# We can Load the pre-trained VGG model
vgg_model = models.vgg16(pretrained=True) # We create an instance of the VGG16 model. The with pretrained=True we can load pre-trained weights (from ImageNet dataset).
# models: models.vgg16, models.vgg19; models.resnet18, models.resnet34, ..., models.resnet152; models.densenet121, models.densenet169; models.inception_v3; models.mobilenet_v2; models.alexnet, models.squeezenet1_1

# vgg_model.classifier[-1] = nn.Linear(vgg_model.classifier[-1].in_features, len(train_data.classes)) # If you need you can add more lines to creates a new fully connected layer 
# With the code vgg_model.classifier[-1].in_features: retrieves the number of input features (or dimensions) of the last layer in the classifier. 

#-----------------Load-------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
vgg_model = vgg_model.to(device)

if os.path.isfile('weights.pth'):    
    vgg_model.load_state_dict(torch.load('weights.pth')) # replace weights from a path
    print("Saved Weights loaded.")

#----------------------------------------

# Define a Loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg_model.parameters(), lr=0.001, momentum=0.9)

# Lists to store the loss values and iterations
loss_values_list = []
iterations = []

# Initialize the plot
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(loss_values_list)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')

# Train the network
for epoch in range(1):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = vgg_model(inputs)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        avg_batch_loss = loss.item() / len(train_loader)
        print('[%d, %0d] loss: %.8f' % (epoch + 1, i + 1, avg_batch_loss))

        loss_values_list.append(avg_batch_loss)
        iterations.append(epoch * len(train_loader) + i)

        # Update the plot
        line1.set_xdata(iterations)
        line1.set_ydata(loss_values_list)
        ax.relim()
        ax.autoscale_view(True,True,True)
        fig.canvas.draw()
        plt.pause(0.01)

print('Saivint weights')
torch.save(vgg_model.state_dict(), 'weights.pth')
print('Finished Training')

# Function to predict the image class
def predict_image(image_path):
    img = Image.open(image_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(img)

    image_tensor = image_tensor.unsqueeze_(0)
    output = vgg_model(image_tensor.to(device))

    _, predicted = torch.max(output, 1)
    class_name = train_data.classes[predicted.item()]
    
    return class_name, img

# Test the trained network on an image 
class_name, img = predict_image("/media/xgang/XGang-1T/Graduation/First_Year/Machine-Learning/WEEK10/flower_dataset/test/Image_5.jpg")

# Plot the image and the predicted class
plt.imshow(img)
plt.title(f"Result of predicted class: {class_name}")
plt.show()

plt.ioff()
plt.show()
