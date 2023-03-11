#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import functions as F
import functions as C
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#-----------------------------------------------------------------------------------------#

path = '../datasets/satellite_imagery/flood_classes/raw_data/'
nonflood = mpimg.imread(path + 'beforeflood.png')
nonflood[np.isnan(nonflood)] = 0
label_per = mpimg.imread(path + 'label_permanence.png')
# plt.imshow(nonflood)

axisY, axisX, _ = nonflood.shape
print(axisY, axisX)
widthX = 64; heightY = 64
for i in range (0, axisY-65, 64):
	for ii in range (0, axisX-65, 64):
		begX = ii; begY = i
		coordinates = str(begY) + '_' + str(begX) 
		# print(coordinates)
		F.save_image_box(label_per, begX, begY, widthX, heightY, nonflood, 0.9, coordinates)