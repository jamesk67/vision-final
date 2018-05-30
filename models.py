import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime
import pdb
import time
import torchvision.models as torchmodels

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        #if not os.path.exists('logs'):
         #   os.makedirs('logs')
        ts = time.time()
        #st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S_log.txt')
        #st = 'logs/' + st
        #self.logFile = open(st, 'w+')

    def log(self, str):
        print(str)
        self.logFile.write(str + '\n')

    def criterion(self):
        return nn.CrossEntropyLoss()
        #return nn.MSELoss()

    def optimizer(self):
        return optim.Adam(self.parameters())
        #return optim.SGD(self.parameters(), lr=0.001)

    def adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.lr  # TODO: Implement decreasing learning rate's rules
        #if (epoch + 1) % 50 == 0:
         #   lr = lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # never used this to make training run faster
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

       


class LazyNet(BaseModel):
    def __init__(self):
        super(LazyNet, self).__init__()
        # TODO: Define model here
        self.fc1 = nn.Linear(32 * 32 * 3, 10);

    def forward(self, x):
        # TODO: Implement forward pass for LazyNet
        x = x.view(-1, 32 * 32 * 3);
        x = self.fc1(x);
        return x

class BoringNet(BaseModel):
    def __init__(self):
        super(BoringNet, self).__init__()
        # TODO: Define model here
        self.fc1 = nn.Linear(32 * 32 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # TODO: Implement forward pass for BoringNet
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CoolNet(BaseModel):
    def __init__(self):
        super(CoolNet, self).__init__()
        # TODO: Define model here
        self.conv1 = nn.Conv2d(3, 27, 3)
        self.conv2 = nn.Conv2d(27, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(7056, 3000)
        self.fc2 = nn.Linear(3000, 1500)
        self.fc3 = nn.Linear(1500, 555)

    def forward(self, x):
        # TODO: Implement forward pass for CoolNet
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], 7056)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
