import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=5)
        self.conv3 = nn.Conv2d(24, 46, kernel_size=5)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(414, 414)
        self.fc2 = nn.Linear(414, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(self.conv3(x),2)
        x = F.max_pool2d(self.drop(x),2)
        x = x.view(-1, 414)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x) 
        x = self.fc3(x)
        return F.log_softmax(x)