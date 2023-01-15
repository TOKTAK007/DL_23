#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import functions as F
import class_functions as C
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
# # import torch
# #-----------------------------------------------------------------------------------------#
# # from sklearn.model_selection import train_test_split
# from torch import nn
# #-----------------------------------------------------------------------------------------#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using PyTorch version:', torch.__version__, ' Device:', device)
# #-----------------------------------------------------------------------------------------#

beforeflood = skio.imread('../datasets/satellite_imagery/beforeflood_kmeanB4B3B2.tif', plugin='tifffile')
flood = skio.imread('../datasets/satellite_imagery/flood_kmeanB4B3B2.tif', plugin='tifffile')
F.composite_bands(beforeflood, 0.9)
F.composite_bands(flood, 0.9)