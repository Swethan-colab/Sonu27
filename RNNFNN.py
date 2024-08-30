
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np
import copy
from sklearn.preprocessing import RobustScaler
# Replace 'path/to/my/file.h5' with the actual path to your .h5 file in your Drive
data_path = '/upb/users/t/tswethan/profiles/unix/ltm/Swethan/training_data_FFNN__.mat'

import h5py

import numpy as np

# Load the .mat file using h5py
mat_data = h5py.File(data_path, 'r')

# Access the input data modules (assuming 'input_data' is a dataset within the file)
input_data = mat_data['Inputs_FFNN2'][:]
output_data = mat_data['Outputs_FFNN2'][:] # Access the data within the dataset and convert to numpy array
print(input_data)
print(output_data)

# Extract the two input columns
plastic_strain = input_data[0, :]  # Assuming the first column is input1
strain = input_data[1, :]  # Assuming the second column is input2

#Extract 3 ouput columns
energy=output_data[0,:]
stress=output_data[1,:]
dissipation_rate=output_data[2,:]

print(plastic_strain)

serial_numbers_i = np.arange(len(plastic_strain))
serial_numbers_o = np.arange(len(energy))
x =(len(plastic_strain))
print(x)
# Plot the first input column against the serial numbers

# Close the h5py file (good practice)
mat_data.close()

def normalize(data):
  """
  Normalizes data to a range of [0, 1].

  Args:
      data (torch.Tensor): The data to be normalized.

  Returns:
      torch.Tensor: The normalized data.
  """
  min_val = torch.min(data)
  max_val = torch.max(data)
  return (data - min_val) / (max_val - min_val)
def z_score(data):
  return (data - data.mean()) / data.std()

from sklearn.model_selection import train_test_split
X = np.column_stack((plastic_strain, strain))  # Combine input features
y = np.column_stack((energy, stress, dissipation_rate))  # Combine output features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Further split training and validation sets (from X_train_val and y_train_val)


X_train = torch.from_numpy(X_train).float()  # Train data

X_test = torch.from_numpy(X_test).float()  # Test data

print(f"")

y_train = torch.from_numpy(y_train).float()  # Train labels

y_test = torch.from_numpy(y_test).float()  # Test labels
# Assuming 'x' and 'y' are your PyTorch tensors
train_dataset = TensorDataset(normalize(X_train), normalize(y_train))

test_dataset = TensorDataset(normalize(X_test), normalize(y_test))

# Create data loaders for each set
train_dataloader = DataLoader(train_dataset,batch_size=1, shuffle=False)

test_dataloader = DataLoader(test_dataset,batch_size=1, shuffle=False)



patience = 5  # Early stopping patience
learning_rate = 0.0008
import torch.nn as nn
def pinn_loss(prediction, target):
   mse_loss = nn.MSELoss()(prediction,target)
   pinn = torch.mean(torch.relu(-prediction[:,2]))

   return mse_loss,pinn



class MyModel(nn.Module):
       def __init__(self):
            super(MyModel, self).__init__()
            self.gru1 = nn.GRU(2, 128, 3, batch_first=False)
            self.gru2 = nn.GRU(input_size=128, hidden_size=64)
            self.fc1 = nn.Linear(64, 32)  # Adjust hidden size as needed
            self.fc2 = nn.Linear(32, 3)

       def forward(self, x):
            x = self.gru1(x)[0]
            x = self.gru2(x)[0]
            x = torch.relu(self.fc1(x))
            out = self.fc2(x)
            return out


# Create the model
model = MyModel()

def train_model(model, train_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adjust optimizer
    criterion = nn.MSELoss()  # Mean Squared Error loss

    training_losses = []  # Store training losses for each epoch
    pinn_losses = []
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Train loop
        total_train_loss = 0
        pinn = 0
        for data, target in train_loader:
            optimizer.zero_grad()  # Reset gradients for each batch

            output = model(data)
            mse_loss, pinn = pinn_loss(output, target)
            total_loss = mse_loss + pinn
            total_loss.backward()
            optimizer.step()

            pinn_losses.append(pinn.item())
            training_losses.append(total_loss.item())  # Append training loss for each batch

            # Calculate average training loss for the epoch
        avg_train_loss = sum(training_losses) / len(training_losses)

        avg_pinn_loss = sum(pinn_losses) / len(pinn_losses)



        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f} , PINN Loss: {avg_pinn_loss:.4f}")




    # Plot training and validation loss (after training)
    df = pd.DataFrame({'Epoch': range(epochs), 'Training Loss': training_losses, 'Pinn': pinn_loss})

    # Save the DataFrame to an Excel file
    df.to_excel('training_history.xlsx', index=False)

    # Load best model weights (implementation not shown here for brevity)
    return model

train_model(model,train_dataloader,30)
torch.save(model.state_dict(), 'model_RNN_FFNN.pth')
