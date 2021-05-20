import numpy as np
import pandas as pd
from skimage.transform import rotate

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

def rotate_image(x):
  rotatedx = np.zeros(x.shape)
  for i in range(x.shape[0]):
    img = x[i].reshape(48, 48)
    rotated = rotate(img, angle=-45, mode='wrap')
    rotatedx[i] = rotated
  return rotatedx

def flip_image(x):
  flippedx = np.zeros(x.shape)
  for i in range(x.shape[0]):
    img = x[i].reshape(48, 48)
    flipped = np.fliplr(img)
    flippedx[i] = flipped
  return flippedx