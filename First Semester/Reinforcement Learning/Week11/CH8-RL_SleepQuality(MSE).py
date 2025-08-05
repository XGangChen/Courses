import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import os

path = os.path.join(os.path.dirname(__file__), "student_sleep_patterns.csv")
dataset = pd.read_csv(path)
print(dataset.head())
Selected_features = torch.tensor(dataset[['Age', 'Sleep_Duration']].values, dtype=torch.float32)
labels = torch.tensor(dataset['Sleep_Quality'].values, dtype=torch.float32).unsqueeze(1)

# lose_points = [[16, 21], [17, 20], [17, 19], [18, 18], [19, 18], [19, 16], [20, 17], [20, 19]]
# win_points = [[21, 20], [21, 21], [22, 20], [22, 22], [23, 21], [19, 20.5]]
# labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.float32).unsqueeze(1) # Labels: 0 for 'Lose', 1 for 'Win' including sample point

# Convert data to a torch tensor
# all_points = lose_points + win_points
# data = torch.tensor(all_points, dtype=torch.float32)


# def data_plot(sample=None, result=None):
#     plt.figure(figsize=(8, 6))
#     plt.scatter(*zip(*lose_points), color='red', label='Lose')
#     plt.scatter(*zip(*win_points), color='blue', label='Win')
    
#     # Plot the sample point if provided
#     if sample is not None:
#         plt.scatter(sample[0][0], sample[0][1], color='yellow', edgecolor='black', s=100, label=f'Sample Point ({sample[0][0]}, {sample[0][1]})')
        
#         # Add the prediction label next to the sample point with a line break and smaller font
#         plt.text(sample[0][0] + 0.5, sample[0][1], f'[{sample[0][0]}, {sample[0][1]}]\n{result}', 
#                  fontsize=10, ha='left', va='center', color='black')
    
#     plt.xlabel(r'$x_1 = $ Number of hours of swimming practices')
#     plt.ylabel(r'$x_2 = $ Number of hours of dryland workouts')
#     plt.title("Example Problem: Winning Swimming Competition")
#     plt.legend(loc='best')
#     plt.show()


# We can define the neural network model in form of a class
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 5)  # Input to hidden layer
        self.dropout = nn.Dropout(p=0.1) # if you want add dropout
        self.layer2 = nn.Linear(5, 1)  # Hidden to output layer
        self.sigmoid = nn.Sigmoid() # This makes sure that data is in between 0 and 1 for Binary Cross-Entropy Loss

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # ReLU activation for hidden layer
        x = self.layer2(x)  # Sigmoid activation for output layer
        x = self.sigmoid(x)  # Sigmoid activation for output layer

        # x = self.sigmoid(self.layer2(torch.relu(self.layer1(x))))  # We can write it compact like this too
        return x
    
# -------------------------------------------------------------------------------------
# We need to initialize the model, loss function, and optimizer
NN_model = SimpleNN()
loss_function = nn.MSELoss()  # Binary Cross-Entropy Loss
optimizer = optim.RMSprop(NN_model.parameters(), lr=0.0002, momentum=0.8)
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

# Testing a sample point as shown in the slides
sample = torch.tensor([[19.5, 20]], dtype=torch.float32)
prediction = NN_model(sample).item()
result = "Win" if prediction > 0.5 else "Lose"

# data_plot(sample, result)
print("Prediction for sample:", result)
