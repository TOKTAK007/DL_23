#-----------------------------------------------------------------------------------------#
import torch
import torch.nn.functional as TF
import numpy as np
#-----------------------------------------------------------------------------------------#
from torch import nn
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import Dataset
from PIL import Image
#-----------------------------------------------------------------------------------------#

class LinearRegressionModel(nn.Module): 
	def __init__(self):
		super().__init__() 
		self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True) 
		self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True) 
	def forward(self, x: torch.Tensor): 
		return self.weights * x + self.bias 

class MLP(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.input_fc = nn.Linear(input_dim, 250)
		self.hidden_fc = nn.Linear(250, 100)
		self.output_fc = nn.Linear(100, output_dim)

	def forward(self, x):
		# x = [batch size, height, width]
		batch_size = x.shape[0]
		x = x.view(batch_size, -1)
		# x = [batch size, height * width]
		h_1 = TF.relu(self.input_fc(x))
		# h_1 = [batch size, 250]
		h_2 = TF.relu(self.hidden_fc(h_1))
		# h_2 = [batch size, 100]
		y_pred = self.output_fc(h_2)
		# y_pred = [batch size, output dim]
		return y_pred, h_2

class LeNet(nn.Module):
	def __init__(self, output_dim):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=1,
							   out_channels=6,
							   kernel_size=5)
		self.conv2 = nn.Conv2d(in_channels=6,
							   out_channels=16,
							   kernel_size=5)
		self.fc_1 = nn.Linear(16 * 4 * 4, 120)
		self.fc_2 = nn.Linear(120, 84)
		self.fc_3 = nn.Linear(84, output_dim)

	def forward(self, x):
		# x = [batch size, 1, 28, 28]
		x = self.conv1(x)
		# x = [batch size, 6, 24, 24]
		x = TF.max_pool2d(x, kernel_size=2)
		# x = [batch size, 6, 12, 12]
		x = TF.relu(x)
		x = self.conv2(x)
		# x = [batch size, 16, 8, 8]
		x = TF.max_pool2d(x, kernel_size=2)
		# x = [batch size, 16, 4, 4]
		x = TF.relu(x)
		x = x.view(x.shape[0], -1)
		# x = [batch size, 16*4*4 = 256]
		h = x
		x = self.fc_1(x)
		# x = [batch size, 120]
		x = TF.relu(x)
		x = self.fc_2(x)
		# x = batch size, 84]
		x = TF.relu(x)
		x = self.fc_3(x)
		# x = [batch size, output dim]
		return x, h

class AlexNet(nn.Module):
	def __init__(self, output_dim):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, 3, 2, 1),  # in_channels, out_channels, kernel_size, stride, padding
			nn.MaxPool2d(2),  # kernel_size
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 192, 3, padding=1),
			nn.MaxPool2d(2),
			nn.ReLU(inplace=True),
			nn.Conv2d(192, 384, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.MaxPool2d(2),
			nn.ReLU(inplace=True)
		)
		self.classifier = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(256 * 2 * 2, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, output_dim),
		)

	def forward(self, x):
		x = self.features(x)
		h = x.view(x.shape[0], -1)
		x = self.classifier(h)
		return x, h

class AlexNet_64(nn.Module):
	def __init__(self, output_dim):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 128, 5, 2, 2),  # in_channels, out_channels, kernel_size, stride, padding
			nn.MaxPool2d(2),  # kernel_size
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 192, 3, padding=1),
			nn.MaxPool2d(2),
			nn.ReLU(inplace=True),
			nn.Conv2d(192, 384, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.MaxPool2d(2),
			nn.ReLU(inplace=True)
		)
		self.classifier = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(256 * 4 * 4, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, output_dim),
		)

	def forward(self, x):
		x = self.features(x)
		h = x.view(x.shape[0], -1)
		x = self.classifier(h)
		return x, h

class VGG(nn.Module):
	def __init__(self, features, output_dim):
		super().__init__()
		self.features = features
		self.avgpool = nn.AdaptiveAvgPool2d(7)
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(4096, output_dim),
		)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		h = x.view(x.shape[0], -1)
		x = self.classifier(h)
		return x, h

class SatelliteDataset(Dataset):
	def __init__(self, image_paths, class_to_idx, transform=None):
		self.image_paths = image_paths
		self.class_to_idx = class_to_idx
		self.transform = transform

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		image_filepath = self.image_paths[idx]
		image = Image.open(image_filepath).convert('RGB')
		label = image_filepath.split('/')[-2]
		label = self.class_to_idx[label]
		image = np.array(image)
		image = torch.from_numpy(image)
		if self.transform is not None:
			image = self.transform(image)
		return image, label

class ResNet(nn.Module):
	def __init__(self, config, output_dim):
		super().__init__()
		block, layers, channels = config
		self.in_channels = channels[0]
		assert len(layers) == len(channels) == 3
		assert all([i == j*2 for i, j in zip(channels[1:], channels[:-1])])
		self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
		self.bn1 = nn.BatchNorm2d(self.in_channels)
		self.relu = nn.ReLU(inplace = True)
		self.layer1 = self.get_resnet_layer(block, layers[0], channels[0])
		self.layer2 = self.get_resnet_layer(block, layers[1], channels[1], stride = 2)
		self.layer3 = self.get_resnet_layer(block, layers[2], channels[2], stride = 2)
		self.avgpool = nn.AdaptiveAvgPool2d((1,1))
		self.fc = nn.Linear(self.in_channels, output_dim)
		
	def get_resnet_layer(self, block, n_blocks, channels, stride = 1):
		layers = []
		if self.in_channels != channels:
			downsample = True
		else:
			downsample = False
		layers.append(block(self.in_channels, channels, stride, downsample))
		for i in range(1, n_blocks):
			layers.append(block(channels, channels))
		self.in_channels = channels
		return nn.Sequential(*layers)
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.avgpool(x)
		h = x.view(x.shape[0], -1)
		x = self.fc(h)
		return x, h

class Identity(nn.Module):
	def __init__(self, f):
		super().__init__()
		self.f = f
		
	def forward(self, x):
		return self.f(x)

class BasicBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace = True)
		if downsample:
			identity_fn = lambda x : TF.pad(x[:, :, ::2, ::2], [0, 0, 0, 0, in_channels // 2, in_channels // 2])
			downsample = Identity(identity_fn)
		else:
			downsample = None
		self.downsample = downsample
		
	def forward(self, x):
		i = x
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.bn2(x)
		if self.downsample is not None:
			i = self.downsample(i)
		x += i
		x = self.relu(x)
		return x