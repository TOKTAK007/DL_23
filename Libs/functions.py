#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
	# plt.imsave(fname='../datasets/satellite_imagery/flood.png', arr=band_stacking, cmap='viridis', format='png')
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

def draw_box(image, clip, begX, begY, widthX, hightY, save_image):
	a = normalized_data(image[:, :, 0], 0, 1)
	b = normalized_data(image[:, :, 1], 0, 1)
	c = normalized_data(image[:, :, 2], 0, 1)
	band_stacking = np.stack((a, b, c), axis=2) # red always first
	pLow, pHigh = np.percentile(band_stacking[~np.isnan(band_stacking)], (clip, 100-clip))
	band_stacking = exposure.rescale_intensity(band_stacking, in_range=(pLow, pHigh))  # type: ignore
	fig, ax = plt.subplots(figsize=(20, 20))
	ax.imshow(band_stacking, cmap='viridis', alpha=0.8)
	rect = patches.Rectangle((begX, begY), widthX, hightY, linewidth=2, edgecolor='blue', facecolor='none')
	ax.add_patch(rect)
	plt.axis('off')
	save_crop = band_stacking[begX:(begX+widthX), begY:(begY+hightY), :]
	plt.imsave(fname=save_image, arr=save_crop, cmap='viridis', format='png')
	plt.show()

def MAE(data_y, model):
	sum = 0.
	for i in range (0, len(data_y)):
		sum += abs(data_y[i] - model[i])
	return sum/len(data_y)

def MSE(data_y, model):
	sum = 0.
	for i in range (0, len(data_y)):
		sum += (data_y[i] - model[i])**2
	return sum/len(data_y)

# https://gist.github.com/sagarmainkar/41d135a04d7d3bc4098f0664fe20cf3c
def  cal_cost(theta, X, y):
	'''
	Calculates the cost for given X and Y. The following shows and example of a single dimensional X
	theta = Vector of thetas 
	X     = Row of X's np.zeros((2,j))
	y     = Actual y's np.zeros((2,1))
	where:
		j is the no of features
	'''
	m = len(y)
	predictions = X.dot(theta)
	cost = (1/2*m) * np.sum(np.square(predictions-y))
	return cost

def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
	'''
	X    = Matrix of X with added bias units
	y    = Vector of Y
	theta=Vector of thetas np.random.randn(j,1)
	learning_rate 
	iterations = no of iterations
	Returns the final theta vector and array of cost history over no of iterations
	'''
	m = len(y)
	cost_history = np.zeros(iterations)
	theta_history = np.zeros((iterations,2))
	for it in range(iterations):
		prediction = np.dot(X,theta)
		theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - y)))
		theta_history[it,:] =theta.T
		cost_history[it]  = cal_cost(theta, X, y)
	return theta, cost_history, theta_history
