import sys
sys.path.append('./Libs') 
import basic_functions as bf
#-----------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#

# NOTE import data
data = pd.read_csv('../datasets/welllog_csv/welllogs.csv')

# NOTE prepare data
data = data[::10] # select every 10 data points 
# customize plot colors
lithocolors = ['#F4D03F',
               '#F5B041',
               '#DC7633',
               '#6E2C00',
               '#1B4F72',
               '#2E86C1',
               '#AED6F1',
               '#A569BD',
               '#196F3D']
# select well log to visualize
x = data['GR']
y = data['ILD_log10']
labels = data['Facies']
# assign label names to match with the lithocolors
lithofacies = ['SS',
               'CSiS',
               'FSiS',
               'SiSh',
               'MS',
               'WS',
               'D',
               'PS',
               'BS']
# bf.scatter_plot(lithocolors, x, y, labels, lithofacies)

# TODO compute K-Means 
print('label list: ', data['Facies'].unique())
print('lable length: ', len(data['Facies'].unique()))
number_of_clusters = len(data['Facies'].unique())
# compute k-means 
# k_data = bf.compute_kmeans(x, y, number_of_clusters-1)
k_data = bf.compute_kmeans(x, y, number_of_clusters)
# fit kmean data into decided classes
facies = bf.find_nearest(k_data)
lithofacies = ['facie 1',
               'facie 2',
               'facie 3',
               'facie 4',
               'facie 5',
               'facie 6',
               'facie 7',
               'facie 8',
               'facie 9']

# NOTE display final result
# plot k-means
bf.scatter_plot(lithocolors, x, y, facies, lithofacies)