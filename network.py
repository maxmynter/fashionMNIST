import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c1 = nn.Conv2d(1, 48, kernel_size=5)
        self.c2 = nn.Conv2d(48,96, kernel_size=5)
        self.c3 = nn.Conv2d(96,80, kernel_size=3)
        self.c4 = nn.Conv2d(80,96, kernel_size=2)
        self.l1 = nn.Linear(96,96)
        self.l2 = nn.Linear(96,96)
        self.l3 = nn.Linear(96,10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.c1(x),2))
        x = F.relu(F.max_pool2d(self.c2(x),2))
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        x = x.view(-1, 96)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(F.dropout(x, training=self.training)))
        x = F.relu(self.l3(x))
        return F.log_softmax(x)