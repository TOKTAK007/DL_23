#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import neural_network_architectures as NNA
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as TF
#-----------------------------------------------------------------------------------------#
from torch import nn
from tqdm import tqdm
from sklearn import metrics
from sklearn import manifold
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import namedtuple
#-----------------------------------------------------------------------------------------#
params = {
	'savefig.dpi': 300,  
	'figure.dpi' : 300,
	'axes.labelsize':10,  
	'axes.titlesize':10,
	'axes.titleweight': 'bold',
	'legend.fontsize': 8,
	'xtick.labelsize':8,
	'ytick.labelsize':8,
	'font.family': 'serif',
}
matplotlib.rcParams.update(params)
#-----------------------------------------------------------------------------------------#

def plot_images(images):
	n_images = len(images)
	rows = int(np.sqrt(n_images))
	cols = int(np.sqrt(n_images))
	fig = plt.figure(figsize=(20, 20))
	for i in range(rows*cols):
		ax = fig.add_subplot(rows, cols, i+1)
		ax.imshow(images[i].view(28, 28).cpu().numpy(), cmap='bone')
		ax.axis('off')
	plt.show()

def calculate_accuracy(y_pred, y):
	top_pred = y_pred.argmax(1, keepdim=True)
	correct = top_pred.eq(y.view_as(top_pred)).sum()
	acc = correct.float() / y.shape[0]
	return acc

def train(model, iterator, optimizer, criterion, device):
	epoch_loss = 0; epoch_acc = 0
	model.train()
	for (x, y) in tqdm(iterator, desc='Training', leave=False):
		x = x.to(device)
		y = y.to(device)
		optimizer.zero_grad()
		y_pred, _ = model(x)
		loss = criterion(y_pred, y)
		acc = calculate_accuracy(y_pred, y)
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()
		epoch_acc += acc.item()
	return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
	epoch_loss = 0; epoch_acc = 0
	model.eval()
	with torch.no_grad():
		for (x, y) in tqdm(iterator, desc='Evaluating', leave=False):
			x = x.to(device)
			y = y.to(device)
			y_pred, _ = model(x)
			loss = criterion(y_pred, y)
			acc = calculate_accuracy(y_pred, y)
			epoch_loss += loss.item()
			epoch_acc += acc.item()
	return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs

def get_predictions(model, iterator, device):
	model.eval()
	images = []; labels = []; probs = []
	with torch.no_grad():
		for (x, y) in iterator:
			x = x.to(device)
			y_pred, _ = model(x)
			y_prob = TF.softmax(y_pred, dim=-1)
			images.append(x.cpu())
			labels.append(y.cpu())
			probs.append(y_prob.cpu())
	images = torch.cat(images, dim=0)
	labels = torch.cat(labels, dim=0)
	probs = torch.cat(probs, dim=0)
	return images, labels, probs

def plot_confusion_matrix(labels, pred_labels):
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	cm = metrics.confusion_matrix(labels, pred_labels)
	cm = metrics.ConfusionMatrixDisplay(cm, display_labels=range(10))
	cm.plot(values_format='d', cmap='Greens', ax=ax)
	plt.show()

def plot_confusion_matrix_CIFAR10(labels, pred_labels, classes):
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	cm = metrics.confusion_matrix(labels, pred_labels)
	cm = metrics.ConfusionMatrixDisplay(cm, display_labels=classes)
	cm.plot(values_format='d', cmap='Greens', ax=ax)
	# plt.savefig('data_out/' + 'CM_resnet' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

def plot_most_incorrect(incorrect, n_images):
	rows = int(np.sqrt(n_images))
	cols = int(np.sqrt(n_images))
	fig = plt.figure()
	for i in range(rows*cols):
		ax = fig.add_subplot(rows, cols, i+1)
		image, true_label, probs = incorrect[i]
		true_prob = probs[true_label]
		incorrect_prob, incorrect_label = torch.max(probs, dim=0)
		ax.imshow(image.view(28, 28).cpu().numpy(), cmap='bone')
		ax.set_title(f'true label: {true_label} ({true_prob:.3f})\n'
					 f'pred label: {incorrect_label} ({incorrect_prob:.3f})')
		ax.axis('off')
	fig.subplots_adjust(hspace=0.5)
	plt.show()

def normalize_image(image):
	image_min = image.min()
	image_max = image.max()
	image.clamp_(min=image_min, max=image_max)
	image.add_(-image_min).div_(image_max - image_min + 1e-5)
	return image

def plot_most_incorrect_CIFAR10(incorrect, classes, n_images, normalize=True):
	rows = int(np.sqrt(n_images))
	cols = int(np.sqrt(n_images))
	fig = plt.figure()
	for i in range(rows*cols):
		ax = fig.add_subplot(rows, cols, i+1)
		image, true_label, probs = incorrect[i]
		image = image.permute(1, 2, 0)
		true_prob = probs[true_label]
		incorrect_prob, incorrect_label = torch.max(probs, dim=0)
		true_class = classes[true_label]
		incorrect_class = classes[incorrect_label]
		if normalize:
			image = normalize_image(image)
		ax.imshow(image.cpu().numpy())
		ax.set_title(f'true label: {true_class} ({true_prob:.3f})\n'
             f'pred label: {incorrect_class} ({incorrect_prob:.3f})',
             fontsize=6)
		ax.axis('off')
	fig.subplots_adjust(hspace=0.6)
	# plt.savefig('data_out/' + 'incorrect_resnet' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

def plot_CIFAR10(images, labels, classes, normalize=False):
	n_images = len(images)
	rows = int(np.sqrt(n_images))
	cols = int(np.sqrt(n_images))
	fig = plt.figure(figsize=(20, 20))
	for i in range(rows*cols):
		ax = fig.add_subplot(rows, cols, i+1)
		image = images[i]
		if normalize:
			image_min = image.min()
			image_max = image.max()
			image.clamp_(min=image_min, max=image_max)
			image.add_(-image_min).div_(image_max - image_min + 1e-5)
		ax.imshow(image.permute(1, 2, 0).cpu().numpy())
		ax.set_title(classes[labels[i]])
		ax.axis('off')
	plt.show()

def get_representations(model, iterator, device):
	model.eval()
	outputs = []
	intermediates = []
	labels = []
	with torch.no_grad():
		for (x, y) in tqdm(iterator):
			x = x.to(device)
			y_pred, h = model(x)
			outputs.append(y_pred.cpu())
			intermediates.append(h.cpu())
			labels.append(y)
	outputs = torch.cat(outputs, dim=0)
	intermediates = torch.cat(intermediates, dim=0)
	labels = torch.cat(labels, dim=0)
	return outputs, intermediates, labels

def plot_representations(data, labels, n_images=None):
	if n_images is not None:
		data = data[:n_images]
		labels = labels[:n_images]
	fig = plt.figure(figsize=(20, 20))
	ax = fig.add_subplot(111)
	scatter = ax.scatter(data[:, 0], data[:, 1], s=60, linewidths=2, edgecolors='black',  c=labels, cmap='tab10')
	handles, labels = scatter.legend_elements()
	ax.legend(handles=handles, labels=labels)
	plt.show()

def plot_representations_CIFAR10(data, labels, n_images, classes):
	if n_images is not None:
		data = data[:n_images]
		labels = labels[:n_images]
	fig = plt.figure(figsize=(20, 20))
	ax = fig.add_subplot(111)
	scatter = ax.scatter(data[:, 0], data[:, 1], s=60, linewidths=2, edgecolors='black',  c=labels, cmap='tab10')
	handles, labels = scatter.legend_elements()
	ax.legend(handles=handles, labels=classes)
	plt.show()

def get_tsne(data, n_components=2, n_images=None):
	if n_images is not None:
		data = data[:n_images]
	tsne = manifold.TSNE(n_components=n_components, random_state=0)
	tsne_data = tsne.fit_transform(data)
	return tsne_data

def get_vgg_layers(config, batch_norm):
	layers = []
	in_channels = 3
	for c in config:
		assert c == 'M' or isinstance(c, int)
		if c == 'M':
			layers += [nn.MaxPool2d(kernel_size=2)]
		else:
			conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = c
	return nn.Sequential(*layers)

def loss_history_plot(history_train, history_valid, model_name):
	axis_x = np.linspace(0, len(history_train), len(history_train))
	plt.plot(axis_x, history_train, linestyle='solid',
			 color='red', linewidth=1, marker='o', ms=5, label='train')
	plt.plot(axis_x, history_valid, linestyle='solid',
			 color='blue', linewidth=1, marker='o', ms=5, label='valid')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend(['train', 'valid'])
	plt.title(model_name + ': ' + 'Accuracy', fontweight='bold')
	# plt.savefig('data_out/' + 'resnet' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

def imshow_numpy_format(train_list):
	plt.subplots(4, 4, figsize=(20, 20))
	for i, k in enumerate(np.random.randint(500, size=16)):
		im = Image.open(train_list[k])
		plt.subplot(4, 4, i+1)
		plt.xlabel(train_list[k].split('/')[-2])
		plt.imshow(np.array(im), cmap='viridis')
		plt.tight_layout()
	plt.show()

def augmentation(meanRGB, stdRGB, train_list, class_to_idx, val_list, test_list):
	# NOTE augment for training and validation data
	train_transforms = transforms.Compose([transforms.ToPILImage(),
										#  transforms.RandomResizedCrop(224),
										 transforms.RandomHorizontalFlip(),
										 transforms.ToTensor(),
										 transforms.Normalize(meanRGB, stdRGB)])
	val_transforms = transforms.Compose([transforms.ToPILImage(),
									#    transforms.CenterCrop(224),
									   transforms.ToTensor(),
									   transforms.Normalize(meanRGB, stdRGB)])
	# NOTE SatelliteDataset class uses for rearrange images and labels that retrive from the input folder
	train_dataset = NNA.SatelliteDataset(train_list, class_to_idx, train_transforms)
	val_dataset = NNA.SatelliteDataset(val_list, class_to_idx, val_transforms)
	test_dataset = NNA.SatelliteDataset(test_list, class_to_idx, val_transforms)
	# NOTE DataLoader, torch module, will rearrange images and labels to fit PyTorch format. batch_size=4 is for viewing only.
	train_iterator = DataLoader(train_dataset, batch_size=4, shuffle=True)
	val_iterator = DataLoader(val_dataset, batch_size=4, shuffle=True)
	test_iterator = DataLoader(test_dataset, batch_size=4, shuffle=False)
	return train_iterator, val_iterator, test_iterator

def quick_show_torch(inp, meanRGB, stdRGB, title=None):
	inp = inp.numpy().transpose((1, 2, 0))
	mean = np.array(meanRGB)
	std = np.array(stdRGB)
	inp = std * inp + mean
	inp = np.clip(inp, 0, 1)
	plt.imshow(inp)
	if title is not None:
		plt.title(title)
	plt.pause(0.001)  # pause a bit so that plots are updated
	plt.show()

def ResNet_achitecture_choices(ResNet_achitecture):
	ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
	if ResNet_achitecture == 'ResNet20':
		n_blocks = [3, 3, 3]
		print('Using ResNet20')
	elif ResNet_achitecture == 'ResNet32':
		n_blocks = [5, 5, 5]
		print('Using ResNet32')
	elif ResNet_achitecture == 'ResNet44':
		n_blocks = [7, 7, 7]
		print('Using ResNet44')
	elif ResNet_achitecture == 'ResNet56':
		n_blocks = [9, 9, 9]
		print('Using ResNet56')
	elif ResNet_achitecture == 'ResNet110':
		n_blocks = [18, 18, 18]
		print('Using ResNet110')
	elif ResNet_achitecture == 'ResNet1202':
		n_blocks = [20, 20, 20]
		print('Using ResNet1202')
	else:
		print('out of ResNet architectures')
	return ResNetConfig(block = NNA.BasicBlock, n_blocks = n_blocks, channels = [16, 32, 64])