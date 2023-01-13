#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import functions as F
import class_functions as C
#-----------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#
from torchvision import datasets, transforms
#-----------------------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', device)
#-----------------------------------------------------------------------------------------#

# NOTE predifined parameters
batch_size = 128 # depends on GPU memory size
learning_rate = 0.01
momentum = 0.5
epochs = 5
