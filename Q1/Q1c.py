import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from ncutil import *

class NeuralNet(nn.Module):
	def __init__(self, n, r):
		super(NeuralNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 64, kernel_size=(3,3), stride=3, padding=0)
		self.bn1 = nn.BatchNorm2d(64)
		self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=(2,2), stride=2, padding=0)
		self.bn2 = nn.BatchNorm2d(128)
		self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)
		self.fc1 = nn.Linear(512, 256)
		self.bn3 = nn.BatchNorm1d(256)
		self.fc2 = nn.Linear(256, r)
		self.bn4 = nn.BatchNorm1d(r)

	def forward(self, x):
		x = self.pool1(F.relu(self.bn1(self.conv1(x))))
		x = self.pool2(F.relu(self.bn2(self.conv2(x))))
		x = x.view(x.shape[0],-1)
		x = F.relu(self.bn3(self.fc1(x)))
		x = self.bn4(self.fc2(x))
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
			if epochs > 10 and error < 1e-5 or epochs > 500:
				converged = True

train_data = sys.argv[1]
test_data = sys.argv[2]
output_file = sys.argv[3]

x_train, y_train = getData(train_data)
x_test, y_test = getData(test_data)

x_train = x_train.reshape(x_train.shape[0], 1, 48, 48)
x_test = x_test.reshape(x_test.shape[0], 1, 48, 48)

tx_train = torch.FloatTensor(x_train.tolist()).cuda()
ty_train = torch.LongTensor(y_train.tolist()).cuda()
tx_test = torch.FloatTensor(x_test.tolist()).cuda()

model = NeuralNet(x_train.shape[1], 7)
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

model.fit(tx_train, ty_train, 100, optimizer, loss_func)

test_pred = model.predict(tx_test)

f = open(output_file, 'w')
for p in test_pred:
	print(int(p), file=f)
f.close()