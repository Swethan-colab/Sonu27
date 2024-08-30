import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np
import copy
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
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






epochs = 1000

patience = 5  # Early stopping patience
learning_rate = 0.0008

def pinn_loss(prediction, target):
   mse_loss = torch.mean((prediction - target) ** 2)
   negative_penalty = torch.mean(torch.relu(-prediction[:,2]))

   return mse_loss, negative_penalty

import torch.nn as nn

class MyModel(nn.Module):
       def __init__(self):
            super(MyModel, self).__init__()
            self.gru1 = nn.GRU(2, 64, 3, batch_first=True)
            self.gru2 = nn.GRU(input_size=64, hidden_size=32)
            self.fc1 = nn.Linear(32, 16)  # Adjust hidden size as needed
            self.fc2 = nn.Linear(16, 3)

       def forward(self, x):
            x = self.gru1(x)[0]
            x = self.gru2(x)[0]
            x = torch.relu(self.fc1(x))
            out = self.fc2(x)
            return out


# Create the model

class PyTorchClassifier(BaseEstimator, ClassifierMixin, nn.Module):
    def __init__(self,model,learning_rate=0.01, optimizer=torch.optim.Adam ):
        super(PyTorchClassifier, self).__init__()  # Call superclass __init__ first
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.model = model

    def fit(self,X_train, y_train):
        X_train = torch.from_numpy(X_train).float()  # Train data

         # Test data



        y_train = torch.from_numpy(y_train).float()  # Train labels

          # Test labels
        # Assuming 'x' and 'y' are your PyTorch tensors
        train_dataset = TensorDataset(normalize(X_train), normalize(y_train))



        # Create data loaders for each set
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)



        # Create model
        self.model = MyModel()


        # Create optimizer
        optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)

        # Training loop

        for data,target in train_dataloader:
            optimizer.zero_grad()
            output = self.model(data)
            mse_loss, pinn = pinn_loss(output, target)
            total_loss = mse_loss + pinn
            total_loss.backward()
            optimizer.step()

        self.classes_ = np.unique(y_train)
        return self

    def predict(self, X_train):
        with torch.no_grad():

            X_train = torch.from_numpy(X_train).float()
            predictions = self.model(X_train)
        return predictions.numpy()


param_grid = {



    'learning_rate': [0.01, 0.001, 0.0001],
    'optimizer':[torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD]


}

# Create GridSearchCV object

model1 = PyTorchClassifier(MyModel)
grid_search = GridSearchCV(estimator=model1, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit GridSearchCV to training data
grid_search.fit(X_train,y_train)

# Get best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_



print("Best Parameters:", best_params)
