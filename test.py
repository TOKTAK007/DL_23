import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

path = '../datasets/satellite_imagery/flood_classes/raw_data/'
nonflood = mpimg.imread(path + 'beforeflood.png')
label_per = mpimg.imread(path + 'label_permanence.png')
# plt.imshow(nonflood)
plt.imshow(label_per[:, :, 0], cmap='gray')
plt.show()