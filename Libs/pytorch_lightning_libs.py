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

import torch.optim as optim

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from types import SimpleNamespace

# pl.seed_everything(42)
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
act_fn_by_name = {
	"tanh": nn.Tanh,
	"relu": nn.ReLU,
	"leakyrelu": nn.LeakyReLU,
	"gelu": nn.GELU
}
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

class DenseLayer(nn.Module):
	
	def __init__(self, c_in, bn_size, growth_rate, act_fn):
		"""
		Inputs:
			c_in - Number of input channels
			bn_size - Bottleneck size (factor of growth rate) for the output of the 1x1 convolution. Typically between 2 and 4.
			growth_rate - Number of output channels of the 3x3 convolution
			act_fn - Activation class constructor (e.g. nn.ReLU)
		"""
		super().__init__()
		self.net = nn.Sequential(
			nn.BatchNorm2d(c_in),
			act_fn(),
			nn.Conv2d(c_in, bn_size * growth_rate, kernel_size=1, bias=False),
			nn.BatchNorm2d(bn_size * growth_rate),
			act_fn(),
			nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
		)

	def forward(self, x):
		out = self.net(x)
		out = torch.cat([out, x], dim=1)
		return out

class DenseBlock(nn.Module):
	
	def __init__(self, c_in, num_layers, bn_size, growth_rate, act_fn):
		"""
		Inputs:
			c_in - Number of input channels
			num_layers - Number of dense layers to apply in the block
			bn_size - Bottleneck size to use in the dense layers
			growth_rate - Growth rate to use in the dense layers
			act_fn - Activation function to use in the dense layers
		"""
		super().__init__()
		layers = []
		for layer_idx in range(num_layers):
			layers.append(
				DenseLayer(c_in=c_in + layer_idx * growth_rate, # Input channels are original plus the feature maps from previous layers
						   bn_size=bn_size,
						   growth_rate=growth_rate,
						   act_fn=act_fn)
			)
		self.block = nn.Sequential(*layers)

	def forward(self, x):
		out = self.block(x)
		return out

class TransitionLayer(nn.Module):
	
	def __init__(self, c_in, c_out, act_fn):
		super().__init__()
		self.transition = nn.Sequential(
			nn.BatchNorm2d(c_in),
			act_fn(),
			nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
			nn.AvgPool2d(kernel_size=2, stride=2) # Average the output for each 2x2 pixel group
		)

	def forward(self, x):
		return self.transition(x)
	
class DenseNet(nn.Module):
	
	def __init__(self, num_classes=10, num_layers=[6,6,6,6], bn_size=2, growth_rate=16, act_fn_name="relu", **kwargs):
		super().__init__()
		self.hparams = SimpleNamespace(num_classes=num_classes,
									   num_layers=num_layers,
									   bn_size=bn_size,
									   growth_rate=growth_rate,
									   act_fn_name=act_fn_name,
									   act_fn=act_fn_by_name[act_fn_name])
		self._create_network()
		self._init_params()

	def _create_network(self):
		c_hidden = self.hparams.growth_rate * self.hparams.bn_size 
		self.input_net = nn.Sequential(
			nn.Conv2d(3, c_hidden, kernel_size=3, padding=1) 
		)

		blocks = []
		for block_idx, num_layers in enumerate(self.hparams.num_layers):
			blocks.append(
				DenseBlock(c_in=c_hidden,
						   num_layers=num_layers,
						   bn_size=self.hparams.bn_size,
						   growth_rate=self.hparams.growth_rate,
						   act_fn=self.hparams.act_fn)
			)
			c_hidden = c_hidden + num_layers * self.hparams.growth_rate # Overall output of the dense block
			if block_idx < len(self.hparams.num_layers)-1: # Don't apply transition layer on last block
				blocks.append(
					TransitionLayer(c_in=c_hidden,
									c_out=c_hidden // 2,
									act_fn=self.hparams.act_fn))
				c_hidden = c_hidden // 2

		self.blocks = nn.Sequential(*blocks)

		# Mapping to classification output
		self.output_net = nn.Sequential(
			nn.BatchNorm2d(c_hidden), # The features have not passed a non-linearity until here.
			self.hparams.act_fn(),
			nn.AdaptiveAvgPool2d((1,1)),
			nn.Flatten(),
			nn.Linear(c_hidden, self.hparams.num_classes)
		)

	def _init_params(self):
		# Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_fn_name)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.input_net(x)
		x = self.blocks(x)
		x = self.output_net(x)
		return x

class PL_Module(pl.LightningModule):
	# act_fn_by_name = {
	# 	"tanh": nn.Tanh,
	# 	"relu": nn.ReLU,
	# 	"leakyrelu": nn.LeakyReLU,
	# 	"gelu": nn.GELU
	# }

	model_dict = {}
	model_dict["DenseNet"] = DenseNet

	def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
		super().__init__()
		self.save_hyperparameters()
		self.model = self.create_model(model_name, model_hparams)
		self.loss_module = nn.CrossEntropyLoss()
		self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

	def forward(self, imgs):
		return self.model(imgs)

	def configure_optimizers(self):
		if self.hparams.optimizer_name == "Adam":
			optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
		elif self.hparams.optimizer_name == "SGD":
			optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
		else:
			assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
		return [optimizer], [scheduler]

	def training_step(self, batch, batch_idx):
		imgs, labels = batch
		preds = self.model(imgs)
		loss = self.loss_module(preds, labels)
		acc = (preds.argmax(dim=-1) == labels).float().mean()

		self.log("train_acc", acc, on_step=False, on_epoch=True)
		self.log("train_loss", loss)
		return loss 

	def validation_step(self, batch, batch_idx):
		imgs, labels = batch
		preds = self.model(imgs).argmax(dim=-1)
		acc = (labels == preds).float().mean()
		self.log("val_acc", acc)

	def test_step(self, batch, batch_idx):
		imgs, labels = batch
		preds = self.model(imgs).argmax(dim=-1)
		acc = (labels == preds).float().mean()
		self.log("test_acc", acc)

	@staticmethod
	def create_model(model_name, model_hparams):
		if model_name in PL_Module.model_dict:
			return PL_Module.model_dict[model_name](**model_hparams)
		else:
			assert False, f"Unknown model name \"{model_name}\". Available models are: {str(PL_Module.model_dict.keys())}"


def train_model(model_name,
				save_name=None,
				CHECKPOINT_PATH=None,
				train_loader=None,
				val_loader=None,
				test_loader=None,
				**kwargs):
	if save_name is None:
		save_name = model_name
	trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name), 
						 max_epochs=5,
						 gpus=1,
						 callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
									LearningRateMonitor("epoch")],                                           # Log learning rate every epoch
						 enable_progress_bar=True)                                                           # Set to False if you do not want a progress bar
	trainer.logger._log_graph = True    
	trainer.logger._default_hp_metric = None 
	pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
	if os.path.isfile(pretrained_filename):
		print(f"Found pretrained model at {pretrained_filename}, loading...")
		model = PL_Module.load_from_checkpoint(pretrained_filename) 
	else:
		model = PL_Module(model_name=model_name, **kwargs)
		trainer.fit(model, train_loader, val_loader)
		model = PL_Module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) 
	val_result = trainer.test(model, val_loader, verbose=False)
	test_result = trainer.test(model, test_loader, verbose=False)
	result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
	return model, result
