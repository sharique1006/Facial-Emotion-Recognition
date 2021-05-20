import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from cutil import *

class VGG(nn.Module):
    def __init__(self, cfg):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

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
        if epochs > 10 and error < 1e-5 or epochs > 100:
          converged = True

train_data = sys.argv[1]
test_data = sys.argv[2]
output_file = sys.argv[3]

x_train, y_train = getData(train_data)
x_test, y_test = getData(test_data)

x_train = x_train.reshape(x_train.shape[0], 1, 48, 48)
x_test = x_test.reshape(x_test.shape[0], 1, 48, 48)

flipped_xtrain = flip_image(x_train)
rotated_xtrain = rotate_image(x_train)

fx_train = np.concatenate((x_train, flipped_xtrain))
fx_train = np.concatenate((fx_train, rotated_xtrain))
fy_train = np.concatenate((y_train, y_train))
fy_train = np.concatenate((fy_train, y_train))

tfx_train = torch.FloatTensor(fx_train.tolist()).cuda()
tfy_train = torch.LongTensor(fy_train.tolist()).cuda()
tx_test = torch.FloatTensor(x_test.tolist()).cuda()

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
model = VGG(cfg)
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

model.fit(tfx_train, tfy_train, 100, optimizer, loss_func)

test_pred = model.predict(tx_test)

f = open(output_file, 'w')
for p in test_pred:
  print(int(p), file=f)
f.close()
