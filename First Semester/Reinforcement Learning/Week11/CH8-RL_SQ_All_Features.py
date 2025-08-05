import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import os
import random


path = os.path.join(os.path.dirname(__file__), "student_sleep_patterns.csv")
dataset = pd.read_csv(path)
print(dataset.info())

# Mapping Gender column
gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
dataset['Gender'] = dataset['Gender'].map(gender_mapping)

# Mapping University_Year column
university_year_mapping = {'1st Year': 0, '2nd Year': 1, '3rd Year': 2, '4th Year': 3}
dataset['University_Year'] = dataset['University_Year'].map(university_year_mapping)

Selected_features = torch.tensor(
    dataset[
        ['Age', 'Gender', 'University_Year', 'Sleep_Duration', 'Study_Hours', 'Screen_Time', 'Caffeine_Intake', 
         'Physical_Activity', 'Sleep_Quality', 'Weekday_Sleep_Start', 'Weekend_Sleep_Start', 'Weekday_Sleep_End', 'Weekend_Sleep_End']
        ].values, dtype=torch.float32)
labels = torch.tensor(dataset['Sleep_Quality'].values, dtype=torch.float32).unsqueeze(1)

# We can define the neural network model in form of a class
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(13, 27)  # Input to hidden layer
        self.layer2 = nn.Linear(27, 9)  # Hidden to output layer
        self.layer3 = nn.Linear(9, 1)
        self.dropout = nn.Dropout(p=0.2) # if you want add dropout
        self.sigmoid = nn.Sigmoid() # This makes sure that data is in between 0 and 1 for Binary Cross-Entropy Loss

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # ReLU activation for hidden layer
        x = torch.relu(self.layer2(x))  # ReLU activation for hidden layer
        x = self.layer3(x)  # Sigmoid activation for output layer
        x = self.sigmoid(x)  # Sigmoid activation for output layer

        # x = self.sigmoid(self.layer2(torch.relu(self.layer1(x))))  # We can write it compact like this too
        return x
    
# -------------------------------------------------------------------------------------
# We need to initialize the model, loss function, and optimizer
NN_model = SimpleNN()
# loss_function = nn.MSELoss()
loss_function = nn.SmoothL1Loss()
# optimizer = torch.optim.RMSprop(NN_model.parameters(), lr=0.0001, momentum=0.8)
# optimizer = torch.optim.Adam(NN_model.parameters(), lr=0.0001)
# optimizer = torch.optim.Adagrad(NN_model.parameters(), lr=0.0001)
optimizer = torch.optim.SGD(NN_model.parameters(), lr=0.0001, momentum=0.8)
loss_values = []

# The main loop for training our model
max_epochs = 10000
for epoch in range(max_epochs):
    optimizer.zero_grad()  # Initialize the gradients as zero
    outputs = NN_model(Selected_features)  # Forward Propagation
    loss = loss_function(outputs, labels)  # Calculate the loss
    loss.backward()  # Backpropagation (compute gradients)
    optimizer.step()  # Update the weights based on gradients

    print(f"Epoch => {epoch}/{max_epochs}, Loss: {loss.item():.4f}")
    loss_values.append(loss.item())

# Plot the loss convergence
plt.figure(figsize=(8, 6))
plt.plot(loss_values, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss value for Sleeping Quality(MSE)')
plt.legend()
plt.show()


# Generate random sample data with 13 features
random_sample = torch.tensor([
    random.randint(18, 25), 
    random.randint(0, 2), 
    random.randint(0, 3), 
    random.uniform(4, 10), 
    random.uniform(0, 8), 
    random.uniform(1, 12), 
    random.randint(0, 5), 
    random.randint(0, 120), 
    random.randint(0, 9), 
    random.uniform(0, 23), 
    random.uniform(0, 23), 
    random.uniform(0, 23), 
    random.uniform(0, 23)
], dtype=torch.float32).unsqueeze(0)
print(random_sample)

prediction = NN_model(random_sample).item()
print(prediction)
