#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import functions as F
import class_functions as C
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import torch
#-----------------------------------------------------------------------------------------#
from sklearn.model_selection import train_test_split
from torch import nn
#-----------------------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', device)
#-----------------------------------------------------------------------------------------#

'''
step 1: create scattering data for a linear equation
'''

number_of_points = 100
weight = 0.7
bias = 3 + torch.randn(number_of_points)*0.9
X = torch.randn(number_of_points)
y = weight*X + bias

'''
step 2: split data for training, validation, and testing
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
# print(len(X_train), len(X_test), len(y_train), len(y_test))
# C.plot_predictions(X_train, y_train, X_test, y_test)

'''
step 3: iniciate linear regression without optimzation. Calling the class function with the default PyTorch module. The input parameters, weight and bias, are predifined, which might yield a bad result.
'''

# model_0 = C.LinearRegressionModel()
# with torch.inference_mode(): 
#     y_preds = model_0(X_test)
# C.plot_predictions(X_train, y_train, X_test, y_test, predictions=y_preds)
# print('mean square error: ', C.MSE(X_test, y_preds))

'''
step 4: train linear regression with optimzation.
'''

# NOTE predefined parameters
model_0 = C.LinearRegressionModel()
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01) 
train_loss_values = []
test_loss_values = []
epoch_count = []
epochs = 1000

for epoch in range(epochs):
	# NOTE training
	model_0.train()
	y_pred = model_0(X_train)
	loss = loss_fn(y_pred, y_train)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	# NOTE validating
	model_0.eval()
	with torch.inference_mode():
		test_pred = model_0(X_test)
		test_loss = loss_fn(test_pred, y_test.type(torch.float))
		# NOTE printing the results during iteration
		if epoch % 10 == 0:
			epoch_count.append(epoch)
			train_loss_values.append(loss.detach().numpy())
			test_loss_values.append(test_loss.detach().numpy())
			print(f'Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ')
# C.loss_curves(epoch_count, train_loss_values, test_loss_values)

# NOTE find our model's learned parameters 
# print('The model learned the following values for weights and bias: ')
# print(model_0.state_dict())
# print('And the original values for weights and bias are: ')
# print(f'weights: {weight}, bias: {bias}')
# # NOTE set the model in evaluation mode
# model_0.eval()
# with torch.inference_mode():
# 	y_preds = model_0(X_test)
# C.plot_predictions(X_train, y_train, X_test, y_test, predictions=y_preds)
# print('mean square error: ', C.MSE(X_test, y_preds))

'''
step 5: save inference models and load the trained models.
'''

# from pathlib import Path

# # 1. Create models directory 
# MODEL_PATH = Path("models")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)

# # 2. Create model save path 
# MODEL_NAME = "01_pytorch_workflow_model_0.pth"
# MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# # 3. Save the model state dict 
# print(f"Saving model to: {MODEL_SAVE_PATH}")
# torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH) 

# # Instantiate a new instance of our model (this will be instantiated with random weights)
# loaded_model_0 = LinearRegressionModel()

# # Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
# loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# # 1. Put the loaded model into evaluation mode
# loaded_model_0.eval()

# # 2. Use the inference mode context manager to make predictions
# with torch.inference_mode():
#     loaded_model_preds = loaded_model_0(X_test) # perform a forward pass on the test data with the loaded model

# # Compare previous model predictions with loaded model predictions (these should be the same)
# y_preds == loaded_model_preds
# print(y_preds)
