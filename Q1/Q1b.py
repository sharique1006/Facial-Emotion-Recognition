import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from ncutil import *

class NeuralNet(nn.Module):
	def __init__(self, n, h, r):
		super(NeuralNet, self).__init__()
		self.l1 = nn.Linear(n, h)
		self.relu = nn.ReLU()
		self.l2 = nn.Linear(h, r)

	def forward(self, x):
		x = self.l1(x)
		x = self.relu(x)
		x = self.l2(x)
		return F.log_softmax(x, dim=1)

	def predict(self, x):
		num_batches = int(x.shape[0]/100)
		pred = np.array([0.])
		for i in range(num_batches):
			x_batch = x[i*100:(i+1)*100]
			pred = np.concatenate((pred, torch.max(self.forward(x_batch).data, 1)[1].cpu().numpy()))
		pred = np.concatenate((pred, torch.max(self.forward(x[num_batches*100:x.shape[0]]), 1)[1].cpu().numpy()))
		return pred[1:]

	def fit(self, x, y, r, optimizer, loss_func):
		num_batches = int(x.shape[0]/r)
		converged = False
		epochs = 0
		prevloss = 0
		error = 0
		while not converged:
			for i in range(num_batches):
				x_batch = x[i*r:(i+1)*r]
				y_batch = y[i*r:(i+1)*r]
				x_var = Variable(x_batch)
				y_var = Variable(y_batch)
				optimizer.zero_grad()
				out = self.forward(x_var)
				loss = loss_func(out, y_var)
				loss.backward()
				optimizer.step()
			error = abs(prevloss - loss.data)
			prevloss = loss.data
			#print('Epoch: {} - Loss: {:.6f} - Error: {:.6f}'.format(epochs, loss.data, error))
			epochs += 1
			del loss
			if epochs > 10 and error < 1e-5 or epochs > 1000:
				converged = True

train_data = sys.argv[1]
test_data = sys.argv[2]
output_file = sys.argv[3]

x_train, y_train = getData(train_data)
x_test, y_test = getData(test_data)

hogx_train = hog_features(x_train)
hogx_test = hog_features(x_test)

htx_train = torch.FloatTensor(hogx_train.tolist()).cuda()
hty_train = torch.LongTensor(y_train.tolist()).cuda()
htx_test = torch.FloatTensor(hogx_test.tolist()).cuda()

hmodel = NeuralNet(hogx_train.shape[1], 100, 7)
hmodel.cuda()
optimizer = optim.SGD(hmodel.parameters(), lr=0.01, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

hmodel.fit(htx_train, hty_train, 100, optimizer, loss_func)

test_pred = hmodel.predict(htx_test)

f = open(output_file, 'w')
for p in test_pred:
	print(int(p), file=f)
f.close()

'''
gaborx_train = gabor_features(x_train)

gtx_train = torch.FloatTensor(gaborx_train.tolist()).cuda()
gty_train = torch.LongTensor(y_train.tolist()).cuda()
gtx_test = torch.FloatTensor(x_test.tolist()).cuda()

gmodel = NeuralNet(x_train.shape[1], 100, 7)
gmodel.cuda()
optimizer = optim.SGD(gmodel.parameters(), lr=0.01, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

gmodel.fit(gtx_train, gty_train, 100, optimizer, loss_func)

test_pred = gmodel.predict(gtx_test)

f = open(output_file, 'w')
for p in test_pred:
	print(int(p), file=f)
f.close()
'''