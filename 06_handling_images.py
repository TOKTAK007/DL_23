#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import functions as F
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
#-----------------------------------------------------------------------------------------#

'''
Task 1: import satellite image as a tiff file from GEE. 
index 0 --> B4 red
index 1 --> B3 green
index 2 --> B2 blue
'''

# beforeflood = skio.imread('../datasets/satellite_imagery/beforeflood_kmeanB4B3B2.tif', plugin='tifffile')
# flood = skio.imread('../datasets/satellite_imagery/flood_kmeanB4B3B2.tif', plugin='tifffile')
# F.imshow_sat(image)
# F.composite_bands(beforeflood, 0.9)
# F.composite_bands(flood, 0.9)

'''
Task 2: import microplastic image as a png file.
'''

mic = skio.imread('../datasets/satellite_imagery/microplastic.png')
mic = mic[:, :, 0:3] # remove the last tensor (not useful) 
# F.imshow_mic(mic)

'''
Task 3: apply kmean
'''

# NOTE compute only the first tensor (red)
mic = mic[:, :, 0]
number_of_classes = 3
k_data = F.compute_kmeans(mic, number_of_classes)
F.imshow_kmean(k_data, number_of_classes)