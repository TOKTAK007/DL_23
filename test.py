#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import torch
#-----------------------------------------------------------------------------------------#
from sklearn.model_selection import train_test_split
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
#-----------------------------------------------------------------------------------------#
print(torch.__version__)
#-----------------------------------------------------------------------------------------#

n = 50
X = np.random.rand(n, 1)
X = torch.from_numpy(X)
noise = np.random.randn(n, 1)
y = 2*X + 3 + noise*0.1
# y = torch.from_numpy(y)

weight = 3 + noise*0.1
bias = 2
# print(X, y)

# plt.scatter(X, y)
# plt.show()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
# print(len(X_train), len(X_test), len(y_train), len(y_test))

# def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
# 	plt.figure(figsize=(20, 20))
# 	# Plot training data in blue
# 	plt.scatter(train_data, train_labels, c="b", s=40, label="Training data")
# 	# Plot test data in green
# 	plt.scatter(test_data, test_labels, c="g", s=40, label="Testing data")
# 	if predictions is not None:
# 		# Plot the predictions in red (predictions were made on the test data)
# 		plt.scatter(test_data, predictions, c="r", s=40, label="Predictions")
# 		# Show the legend
# 		plt.legend(prop={"size": 14});
# 	plt.show()

# # plot_predictions()

# class LinearRegressionModel(nn.Module): 
# 	def __init__(self):
# 		super().__init__() 
# 		self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True) 
# 		self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
# 	def forward(self, x: torch.Tensor): 
# 		return self.weights * x + self.bias 
# 	# def forward(self, x: torch.Tensor): 
# 	# 	return self.weights * x + self.bias 
# 	# def forward(self, x: Tensor.detach().np()): 
# 	# 	return self.weights * x + self.bias 
# 	# .detach().numpy()
# 	# tensor1 = tensor1.detach().numpy()


# torch.manual_seed(42)
# model_0 = LinearRegressionModel()
# # print(list(model_0.parameters()))
# # print(model_0.state_dict())

# # Make predictions with model
# with torch.inference_mode(): 
# 	y_preds = model_0(X_test)

# # Note: in older PyTorch code you might also see torch.no_grad()
# # with torch.no_grad():
# #   y_preds = model_0(X_test)

# # Check the predictions
# print(f"Number of testing samples: {len(X_test)}") 
# print(f"Number of predictions made: {len(y_preds)}")
# print(f"Predicted values:\n{y_preds}")

# # plot_predictions(predictions=y_preds)

# # Create the loss function
# loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# # Create the optimizer
# optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
# 							lr=0.01) # learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))

# torch.manual_seed(42)

# # Set the number of epochs (how many times the model will pass over the training data)
# epochs = 500

# # Create empty loss lists to track values
# train_loss_values = []
# test_loss_values = []
# epoch_count = []

# for epoch in range(epochs):
# 	### Training

# 	# Put model in training mode (this is the default state of a model)
# 	model_0.train()

# 	# 1. Forward pass on train data using the forward() method inside 
# 	y_pred = model_0(X_train)
# 	# print(y_pred)

# 	# 2. Calculate the loss (how different are our models predictions to the ground truth)
# 	loss = loss_fn(y_pred, y_train)

# 	# 3. Zero grad of the optimizer
# 	optimizer.zero_grad()

# 	# 4. Loss backwards
# 	loss.backward()

# 	# 5. Progress the optimizer
# 	optimizer.step()

# 	### Testing

# 	# Put the model in evaluation mode
# 	model_0.eval()

# 	with torch.inference_mode():
# 		# 1. Forward pass on test data
# 		test_pred = model_0(X_test)
# 		# 2. Caculate loss on test data
# 		test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type
# 		# Print out what's happening
# 		if epoch % 10 == 0:
# 			epoch_count.append(epoch)
# 			train_loss_values.append(loss.detach().numpy())
# 			test_loss_values.append(test_loss.detach().numpy())
# 			print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

# # Plot the loss curves
# plt.plot(epoch_count, train_loss_values, label="Train loss")
# plt.plot(epoch_count, test_loss_values, label="Test loss")
# plt.title("Training and test loss curves")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()

# # Find our model's learned parameters
# print("The model learned the following values for weights and bias:")
# print(model_0.state_dict())
# print("\nAnd the original values for weights and bias are:")
# print(f"weights: {weight}, bias: {bias}")

# # 1. Set the model in evaluation mode
# model_0.eval()

# # 2. Setup the inference mode context manager
# with torch.inference_mode():
# 	# 3. Make sure the calculations are done with the model and data on the same device
# 	# in our case, we haven't setup device-agnostic code yet so our data and model are
# 	# on the CPU by default.
# 	# model_0.to(device)
# 	# X_test = X_test.to(device)
# 	y_preds = model_0(X_test)
# # y_preds

# plot_predictions(predictions=y_preds)




