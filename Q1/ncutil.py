import numpy as np
import pandas as pd
from skimage.filters import gabor
from skimage.feature import hog

def getData(file):
	data = pd.read_csv(file, header=None)
	x = data.iloc[:,1:].to_numpy()
	y = data.iloc[:,0].to_numpy()
	y = np.array([int(i) for i in y])
	return x, y
 
def accuracy(y, pred):
	acc = 0
	for i in range(len(y)):
		if pred[i] == y[i]:
			acc += 1
	acc = acc/len(y)
	return acc

def gabor_features(x):
	gaborx = np.zeros(x.shape)
	for i in range(len(x)):
		real,_ = gabor(x[i].reshape((48,48)), frequency=0.6)
		gaborx[i] = real.reshape((1,2304))
	return gaborx

def hog_features(x):
	hogx = np.zeros((x.shape[0], 1296))
	for i in range(x.shape[0]):
		hogx[i] = hog(x[i].reshape(48, 48))
	return hogx