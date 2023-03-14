#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
import pytorch_lightning_libs as PLL
import neural_network_architectures as NNA
#-----------------------------------------------------------------------------------------#
import os
import glob
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
# import time
#-----------------------------------------------------------------------------------------#
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from types import SimpleNamespace
from torchvision import transforms
# from torchsummary import summary
#-----------------------------------------------------------------------------------------#

'''
Step 0: Predefined Parameters.
'''

DATASET_PATH = '../larger_than_50_MB/datasets'
train_size = 0.7; val_size = 0.2; test_size = 0.1
seed = 101
num_workers = 24 # CPU
batch_size = 32
model_name = 'GoogleNet'
CHECKPOINT_PATH = '../larger_than_50_MB/save_trained_model/' + model_name + '.ckpt'
output_dim = 2
image_dim_1 = 32; image_dim_2 = 32
epochs = 10
image2plot = 25
device = PLL.set_seed(seed)

'''
Step 1: Splitting the Dataset and Viewing Images. 
'''

class_names = os.listdir(DATASET_PATH)
print('class names: ', class_names)
num_class = len(class_names)
image_files=glob.glob(DATASET_PATH + '/*/*.png', recursive=True)
print('total images in: ', DATASET_PATH, ' is ', len(image_files))
idx_to_class = {i:j for i, j in enumerate(class_names)}
class_to_idx = {value:key for key, value in idx_to_class.items()}
train_idx, test_idx, val_idx = random_split(image_files, [train_size, val_size, test_size])
train_list=[image_files[i] for i in train_idx.indices]
val_list=[image_files[i] for i in test_idx.indices]
test_list=[image_files[i] for i in val_idx.indices]
print('number of training images: ', len(train_list),
	'\nnumber of val images: ', len(val_list),
	'\nnumber of test images: ', len(test_list))
# PLL.view_images(train_list, num_class)

'''
Step 2: Data Preprocessing
'''

mean, std = PLL.means_std(train_list)
flip = transforms.RandomHorizontalFlip()
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean, std)
transform_train = transforms.Compose([flip, to_tensor, normalize])
transform_val = transforms.Compose([to_tensor, normalize])
transform_test = transforms.Compose([to_tensor, normalize])
train_dataset = PLL.SatelliteDataset(train_list, class_to_idx, transform_train)
val_dataset = PLL.SatelliteDataset(val_list, class_to_idx, transform_val)
test_dataset = PLL.SatelliteDataset(test_list, class_to_idx, transform_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size,
	shuffle=True, drop_last=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
	shuffle=False, drop_last=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
	shuffle=False, drop_last=False, num_workers=num_workers)
# PLL.view_images_from_loader(train_loader, num_class, class_names, mean=mean, std=std)

class SATTELLITEModule(pl.LightningModule):
	def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
		super().__init__()
		self.save_hyperparameters()
		self.model = create_model(model_name, model_hparams)
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
		# self.log("val_acc", acc)
		self.log("val_acc", acc, on_step=False, on_epoch=True, sync_dist=True)

	def test_step(self, batch, batch_idx):
		imgs, labels = batch
		preds = self.model(imgs).argmax(dim=-1)
		acc = (labels == preds).float().mean()
		self.log("test_acc", acc)

model_dict = {}

def create_model(model_name, model_hparams):
	if model_name in model_dict:
		return model_dict[model_name](**model_hparams)
	else:
		assert False, (f'Unknown model name "{model_name}". '
					   f'Available models are: {str(model_dict.keys())}')

act_fn_by_name = {"tanh": nn.Tanh,
				  "relu": nn.ReLU,
				  "leakyrelu": nn.LeakyReLU,
				  "gelu": nn.GELU}

# def train_model(model_name, save_name=None, **kwargs):
#     if save_name is None:
#         save_name = model_name
#     trainer = pl.Trainer(
#         default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),  # Where to save models
#         gpus=1 if str(device) == "cuda:0" else 0,
#         max_epochs=10,
#         callbacks=[
#             ModelCheckpoint(
#                 save_weights_only=True, mode="max", monitor="val_acc"
#             ),  
#             LearningRateMonitor("epoch"),
#         ],  
#     )  
#     trainer.logger._log_graph = True  
#     trainer.logger._default_hp_metric = None  
#     pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
#     if os.path.isfile(pretrained_filename):
#         print(f"Found pretrained model at {pretrained_filename}, loading...")
#         model = SATTELLITEModule.load_from_checkpoint(pretrained_filename)
#     else:
#         pl.seed_everything(42)  
#         model = SATTELLITEModule(model_name=model_name, **kwargs)
#         trainer.fit(model, train_loader, val_loader)
#         model = SATTELLITEModule.load_from_checkpoint(
#             trainer.checkpoint_callback.best_model_path
#         )  
#     val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
#     test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
#     result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
#     return model, result

def train_model(model_name, save_name=None, **kwargs):
# def train_model(model_name, save_name=None):
	if save_name is None:
		save_name = model_name
	trainer = pl.Trainer(
		max_epochs=10,
		accelerator="gpu", devices=2, strategy="ddp",
		default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),  # Where to save models
		callbacks=[
			ModelCheckpoint(
				save_weights_only=True, mode="max", monitor="val_acc"
			),  
			LearningRateMonitor("epoch"),
		],  
	)  
	trainer.logger._log_graph = True  
	trainer.logger._default_hp_metric = None  
	pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
	if os.path.isfile(pretrained_filename):
		print(f"Found pretrained model at {pretrained_filename}, loading...")
		model = SATTELLITEModule.load_from_checkpoint(pretrained_filename)
	else:
		pl.seed_everything(42)  
		model = SATTELLITEModule(model_name=model_name, **kwargs)
		trainer.fit(model, train_loader, val_loader)
		model = SATTELLITEModule.load_from_checkpoint(
			trainer.checkpoint_callback.best_model_path
		)  
	val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
	test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
	result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
	return model, result

class InceptionBlock(nn.Module):
	def __init__(self, c_in, c_red: dict, c_out: dict, act_fn):
		super().__init__()
		self.conv_1x1 = nn.Sequential(
			nn.Conv2d(c_in, c_out["1x1"], kernel_size=1), nn.BatchNorm2d(c_out["1x1"]), act_fn()
		)
		self.conv_3x3 = nn.Sequential(
			nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
			nn.BatchNorm2d(c_red["3x3"]),
			act_fn(),
			nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
			nn.BatchNorm2d(c_out["3x3"]),
			act_fn(),
		)
		self.conv_5x5 = nn.Sequential(
			nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
			nn.BatchNorm2d(c_red["5x5"]),
			act_fn(),
			nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
			nn.BatchNorm2d(c_out["5x5"]),
			act_fn(),
		)
		self.max_pool = nn.Sequential(
			nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
			nn.Conv2d(c_in, c_out["max"], kernel_size=1),
			nn.BatchNorm2d(c_out["max"]),
			act_fn(),
		)

	def forward(self, x):
		x_1x1 = self.conv_1x1(x)
		x_3x3 = self.conv_3x3(x)
		x_5x5 = self.conv_5x5(x)
		x_max = self.max_pool(x)
		x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
		return x_out

class GoogleNet(nn.Module):
	def __init__(self, num_classes=10, act_fn_name="relu", **kwargs):
		super().__init__()
		self.hparams = SimpleNamespace(
			num_classes=num_classes, act_fn_name=act_fn_name, act_fn=act_fn_by_name[act_fn_name]
		)
		self._create_network()
		self._init_params()

	def _create_network(self):
		self.input_net = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), self.hparams.act_fn()
		)
		self.inception_blocks = nn.Sequential(
			InceptionBlock(
				64,
				c_red={"3x3": 32, "5x5": 16},
				c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
				act_fn=self.hparams.act_fn,
			),
			InceptionBlock(
				64,
				c_red={"3x3": 32, "5x5": 16},
				c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
				act_fn=self.hparams.act_fn,
			),
			nn.MaxPool2d(3, stride=2, padding=1),  # 32x32 => 16x16
			InceptionBlock(
				96,
				c_red={"3x3": 32, "5x5": 16},
				c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
				act_fn=self.hparams.act_fn,
			),
			InceptionBlock(
				96,
				c_red={"3x3": 32, "5x5": 16},
				c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
				act_fn=self.hparams.act_fn,
			),
			InceptionBlock(
				96,
				c_red={"3x3": 32, "5x5": 16},
				c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
				act_fn=self.hparams.act_fn,
			),
			InceptionBlock(
				96,
				c_red={"3x3": 32, "5x5": 16},
				c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24},
				act_fn=self.hparams.act_fn,
			),
			nn.MaxPool2d(3, stride=2, padding=1),  # 16x16 => 8x8
			InceptionBlock(
				128,
				c_red={"3x3": 48, "5x5": 16},
				c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
				act_fn=self.hparams.act_fn,
			),
			InceptionBlock(
				128,
				c_red={"3x3": 48, "5x5": 16},
				c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
				act_fn=self.hparams.act_fn,
			),
		)
		self.output_net = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(128, self.hparams.num_classes)
		)

	def _init_params(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_fn_name)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.input_net(x)
		x = self.inception_blocks(x)
		x = self.output_net(x)
		return x

model_dict["GoogleNet"] = GoogleNet

googlenet_model, googlenet_results = train_model(
	model_name="GoogleNet",
	model_hparams={"num_classes": 2, "act_fn_name": "relu"},
	optimizer_name="Adam",
	optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
)