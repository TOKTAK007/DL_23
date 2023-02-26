#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#
from scipy.stats import linregress
#-----------------------------------------------------------------------------------------#
params = {
	'savefig.dpi': 300,  
	'figure.dpi' : 300,
	'axes.labelsize':11,  
	'axes.titlesize':11,
	'axes.titleweight': 'bold',
	'legend.fontsize': 11,  # was 10
	'xtick.labelsize':11,
	'ytick.labelsize':11,
	'font.family': 'serif',
}
matplotlib.rcParams.update(params)
#-----------------------------------------------------------------------------------------#

'''
Python + Numpy
'''

# np.random.seed(0)
# x = np.random.randn(50)
# noise = np.random.randn(50)
# y = 2*x + 5 + noise

# slope = 3; intercept = 8
# yy = slope*x + intercept

# mse_loss = np.mean(np.square(yy - y)) / ((np.max(y) - np.min(y))**2)

# # Calculate the linear regression fit
# slope, intercept, r_value, p_value, std_err = linregress(x, y)
# linear_fn = slope*x + intercept
# regress_loss = np.mean(np.square(linear_fn - y)) / ((np.max(y) - np.min(y))**2)

# plt.figure(figsize=(10, 10))
# plt.scatter(x, y, s=10, color='orange', marker='o', edgecolor='black')
# plt.plot(x, yy, 'g', lw=2, alpha=0.5, label='Quadratic Fit')
# plt.plot(x, linear_fn, 'b', lw=2, alpha=0.5, label='Linear Regression Fit')
# plt.legend()
# plt.xlabel('x'); plt.ylabel('y')
# plt.title(f'Manual: {mse_loss:.3f}, Regression: {regress_loss:.3f}')
# plt.show()

'''
PyTorch
'''

# import torch

# torch.manual_seed(0)
# x = torch.randn(50)
# noise = torch.randn(50)
# y = 2*x + 5 + noise

# slope = 3; intercept = 8
# yy = slope*x + intercept

# manual_loss = torch.mean((yy - y)**2) / ((torch.max(y) - torch.min(y))**2)

# X = torch.stack((x, torch.ones_like(x)), dim=1)
# coef = torch.linalg.lstsq(X, y).solution
# # NOTE The @ operator performs matrix multiplication between X and coef
# linear_fn = X @ coef
# regress_loss = torch.mean((linear_fn - y)**2) / ((torch.max(y) - torch.min(y))**2)

# plt.figure(figsize=(10, 10))
# plt.scatter(x, y, s=10, color='orange', marker='o', edgecolor='black')
# plt.plot(x, yy, 'g', lw=2, alpha=0.5, label='Quadratic Fit')
# plt.plot(x, linear_fn, 'b', lw=2, alpha=0.5, label='Linear Regression Fit')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title(f'Manual: {manual_loss:.3f}, Regression: {regress_loss:.3f}')
# plt.legend()
# plt.show()

'''
PyTorch Class
'''

import torch
from torch.utils.data import Dataset

class SyntheticData(Dataset):
	def __init__(self):
		torch.manual_seed(0)
		self.x = torch.randn(50)
		self.noise = torch.randn(50)
		self.y = 2*self.x + 5 + self.noise

	# NOTE In case we plan to batch a datasets in GPU, not for now.
	# def __len__(self):
	# 	return len(self.x)
	# def __getitem__(self, idx):
	# 	return self.x[idx], self.y[idx]

	def plot_fit(self, slope, intercept):
		yy = slope*self.x + intercept
		manual_loss = torch.mean((yy - self.y)**2) / ((torch.max(self.y) - torch.min(self.y))**2)
		X = torch.stack((self.x, torch.ones_like(self.x)), dim=1)
		coef = torch.linalg.lstsq(X, self.y).solution
		linear_fn = X @ coef
		regress_loss = torch.mean((linear_fn - self.y)**2) / ((torch.max(self.y) - torch.min(self.y))**2)
		print('This time we use class')

		plt.figure(figsize=(10, 10))
		plt.scatter(self.x, self.y, s=10, color='orange', marker='o', edgecolor='black')
		plt.plot(self.x, yy, 'g', lw=2, alpha=0.5, label='Quadratic Fit')
		plt.plot(self.x, linear_fn, 'b', lw=2, alpha=0.5, label='Linear Regression Fit')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.title(f'Manual: {manual_loss:.3f}, Regression: {regress_loss:.3f}')
		plt.legend()
		plt.show()

data = SyntheticData()
data.plot_fit(slope=3, intercept=8)