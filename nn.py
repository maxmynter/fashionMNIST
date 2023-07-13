from fashionData import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mlflow import log_metric, log_param, log_params

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_epochs=2
batch_size = 100
loss_criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
log_params({"Epochs": n_epochs, 
            "Batch Size":batch_size, 
            "loss_criterion": loss_criterion._get_name(),
            'learning_rate':learning_rate})


train = FashionData('data/fashion-mnist_train.csv')
test = FashionData('data/fashion-mnist_test.csv')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

network = Net()
log_param("Network", network)

optimizer = optim.SGD(network.parameters(), lr=learning_rate)





train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)

n_total_steps = len(train_loader)

for epoch in range(n_epochs):
    for i, (labels, images) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass and loss calculation
        outputs = network(images)
        loss = loss_criterion(outputs, labels)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
log_metric("Final Loss", loss.item())

test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=True)


with torch.no_grad():
    n_correct = 0
    n_samples = len(test_loader.dataset)
    for labels, images in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = network(images)

        # max returns (output_value ,index)
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

    acc = n_correct / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {100*acc} %')

log_metric("Accuracy on Test Sets", acc)