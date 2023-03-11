#-----------------------------------------------------------------------------------------#
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn
import os
import pytorch_lightning as pl

from PIL import Image
from skimage import io
from sklearn.preprocessing import StandardScaler
# from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset
# from torchvision import transforms
pl.seed_everything(42)
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

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('Using PyTorch version:', torch.__version__, ' Device:', device)
	return device

def view_images(train_list, num_class):
	num_rows = 5; num_cols = 5
	num_images = num_rows * num_cols
	fig, axes = plt.subplots(num_rows, num_cols)
	displayed_classes = set()
	while len(displayed_classes) < num_class:
		random_images = random.sample(train_list, num_images)
		for i, ax in enumerate(axes.flatten()):
			img_path = random_images[i]
			img = Image.open(img_path)
			class_name = img_path.split('/')[-2]
			ax.imshow(img)
			ax.set_title(f'{class_name}')
			ax.axis('off')
			displayed_classes.add(class_name)
	plt.tight_layout()
	plt.show()

def view_images_from_loader(loader, num_class, class_names, mean=None, std=None):
	num_rows = 5; num_cols = 5
	num_images = num_rows * num_cols
	fig, axes = plt.subplots(num_rows, num_cols)
	displayed_classes = set()
	while len(displayed_classes) < num_class:
		data_iter = iter(loader)
		images, labels = next(data_iter)
		if mean is not None and std is not None:
			images = images.clone().detach().cpu()
			for i in range(3):
				images[:, i, :, :] = images[:, i, :, :] * std[i] + mean[i]
			images = torch.clamp(images, 0, 1)
		for i, ax in enumerate(axes.flatten()):
			img = images[i].permute(1, 2, 0).numpy()
			class_name = class_names[labels[i]]
			ax.imshow(img)
			ax.set_title(f'{class_name}')
			ax.axis('off')
			displayed_classes.add(class_name)
	plt.tight_layout()
	plt.show()

def means_std(train_list):
    train_data = []
    for img_path in train_list:
        img = io.imread(img_path)
        img = img / 255.0  # limit value to be between 0 and 1
        train_data.append(img)
    train_data = np.array(train_data)
    train_data = np.transpose(train_data, (0, 3, 1, 2))  # fit PyTorch format
    train_data_flat = train_data.reshape(train_data.shape[0], -1)
    scaler = StandardScaler()
    scaler.fit(train_data_flat)
    mean = scaler.mean_.reshape(3, -1).mean(axis=1)
    std = scaler.scale_.reshape(3, -1).std(axis=1)
    print('mean: ', mean)
    print('standard deviation: ', std)
    return mean, std

class SatelliteDataset(Dataset):
	def __init__(self, image_paths, class_to_idx, transform=None):
		self.image_paths = image_paths
		self.class_to_idx = class_to_idx
		self.transform = transform

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		image_filepath = self.image_paths[idx]
		image = Image.open(image_filepath)
		label = image_filepath.split('/')[-2]
		label = self.class_to_idx[label]
		if self.transform is not None:
			image = self.transform(image)
		return image, label