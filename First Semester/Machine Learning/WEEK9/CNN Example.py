import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import math 
from PIL import Image
import numpy as np

# we can apply multiple transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),

    # transforms.Grayscale(), # in case we want to use Grayscale
    # transforms.Normalize(mean=[0.5], std=[0.5]), # in case we want to use Grayscale

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # in case of RGB
    # we normaliz to speed up training and leads to faster convergence
    # these numbers commonly used for pre-trained models on ImageNet data, if you dont know add 0.5
    # For each seperate color calculated as the mean and standard deviation of all pixel values in the ImageNet dataset
])

# Load the training data
train_data = datasets.ImageFolder(root='./flower_images/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True) # To make it easy to load our training data in batches during training
# Example Dataset: https://www.kaggle.com/datasets/kausthubkannan/5-flower-types-classification-dataset

class DN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)  # 3 Number of input channels (if grayscale change to 1), 16 is  number of output channels or filters  
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)  
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2) 
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)

        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(w))) 
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(h)))  

        print(convw,convh)

        linear_input_size = convw * convh * 32 # The final convolutional layer (self.conv3) has 32 output channels (filters or kernels), so we have depth of 32
        self.head = nn.Linear(linear_input_size, outputs)

    def conv2d_size_out(self, size, kernel_size=5, stride=2, padding=2): 
            return (size + 2*padding - (kernel_size - 1) - 1) // (stride + 1)
            
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # print("Shape before linear layer: ", x.shape)  # Add this line
        return self.head(x.view(x.size(0), -1)) # reshaping operation to flatten the tensor into a 1-dimensional vector.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # to check ig GPU is available
print("Device:", device)

# We calculate the image size from one sample to calculate our convolution layers sizes
image, _ = train_data[0]
image_size = image.size()
h, w = image_size[1], image_size[2]

# We create an object from class (Initialize the CNN), then move to device (GPU, or CPU)
deepnet = DN(h, w, len(train_data.classes)).to(device)


# Define a Loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(deepnet.parameters(), lr=0.001, momentum=0.9)

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
for epoch in range(10): # loop for training epochs 
    for i, data in enumerate(train_loader, 0): # loop for each batches of data for training the neural network (we perform computations on each image separately, and we update the network's parameters based on the accumulated gradients of one batch) - note: batch dimension allows for efficient parallel processing 
        inputs, labels = data[0].to(device), data[1].to(device) # first element of tuple data[0] is the inputs and the second element data[1] is the labels.
        optimizer.zero_grad()
        outputs = deepnet(inputs) # executes the forward pass to get outputs
        loss = loss_function(outputs, labels.to(device)) # compute the loss by comparing the outputs with the labels

        loss.backward() # compute gradients
        optimizer.step() # updating the model's parameters
        
        avg_batch_loss=loss.item() / len(train_loader)  #  average loss per image in one batch size of taining (loss.item() is loss for the entire batch)
        print('[%d, %0d] loss: %.8f' % (epoch + 1, i + 1, avg_batch_loss))
        
        loss_values_list.append(avg_batch_loss) # to plot
        iterations.append(epoch * len(train_loader) + i) # to plot
        
        # Update the plot
        line1.set_xdata(iterations)
        line1.set_ydata(loss_values_list)
        ax.relim() # update the data limits of the plot 
        ax.autoscale_view(True,True,True)
        fig.canvas.draw() # update the plot after modifying the data
        plt.pause(0.01)

print('Finished Training')

#--------------------------- To pridict and plot a figure --------------------
def predict_image(image_path):
    img = Image.open(image_path) # open the image file

    transform = transforms.Compose([      #Apply the same transformations as we did for the training images
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(img)

    image_tensor = image_tensor.unsqueeze_(0) # convert from (Channels, Height, and Width),  (batch size, Channels, Height, and Width). where batch size = 1 here
    output = deepnet(image_tensor.to(device)) # pass input to our model

    _, predicted = torch.max(output, 1) # To get probabilities, we run a softmax on it

    class_name = train_data.classes[predicted.item()] # We get the class name from the index
    
    return class_name, img

# Test the trained network on an image 
class_name, img = predict_image("./flower_images/test/sample.jpg") # note create the test folder and put an image with name sample.jpg for this part

# Plotting the image and the predicted class
plt.imshow(img)
plt.title(f"Result of predicted class: {class_name}")
plt.show()

plt.ioff()
plt.show()
