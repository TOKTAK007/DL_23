#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#
from skimage import exposure 
from sklearn.cluster import KMeans
#-----------------------------------------------------------------------------------------#
plt.rcParams.update({'font.size': 22})
#-----------------------------------------------------------------------------------------#

def ellipse(b, x, a):
	p1 = pow(x, 2)/pow(a, 2)
	p2 = np.sqrt(1000 - p1)
	y = b*p2
	return y

def imshow_sat(image):
	plt.figure(figsize=(20, 20))
	plt.imshow(image[:, :, 0], cmap='terrain') # plot only the first index
	plt.title('B4', fontweight='bold')
	plt.xlabel('pixel-X'); plt.ylabel('pixel-Y')
	plt.show()

def imshow_mic(image):
	plt.figure(figsize=(20, 20))
	plt.imshow(image)
	plt.title('microplastic', fontweight='bold')
	plt.xlabel('pixel-X'); plt.ylabel('pixel-Y')
	print('tensor dimensions: ', image.shape)
	plt.show()

def normalized_data(y, lowest_value, highest_value):
	y = (y - y.min()) / (y.max() - y.min())
	return y * (highest_value - lowest_value) + lowest_value

def composite_bands(image, clip):
	a = normalized_data(image[:, :, 0], 0, 1)
	b = normalized_data(image[:, :, 1], 0, 1)
	c = normalized_data(image[:, :, 2], 0, 1)
	band_stacking = np.stack((a, b, c), axis=2) # red always first
	pLow, pHigh = np.percentile(band_stacking[~np.isnan(band_stacking)], (clip, 100-clip))
	band_stacking = exposure.rescale_intensity(band_stacking, in_range=(pLow, pHigh))  # type: ignore
	plt.figure(figsize=(20, 20))
	plt.imshow(band_stacking, cmap='viridis', alpha=0.8)
	plt.axis('off')
	plt.show()

def compute_kmeans(image, number_of_classes):
	vector_data = image.reshape(-1, 1) 
	random_centroid = 42 
	kmeans = KMeans(n_clusters = number_of_classes, random_state = random_centroid, n_init='auto').fit(vector_data)
	kmeans = kmeans.cluster_centers_[kmeans.labels_]
	return kmeans.reshape(image.shape)

def imshow_kmean(data, number_of_classes):
	plt.figure(figsize=(20, 20))
	cmap = plt.get_cmap('rainbow', number_of_classes)
	plt.imshow(data, cmap=cmap)
	plt.title('K-means', fontweight='bold')
	plt.xlabel('pixel-X'); plt.ylabel('pixel-Y')
	plt.colorbar()
	plt.show()