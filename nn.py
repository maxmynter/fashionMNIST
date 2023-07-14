from fashionData import *
from network import Net
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mlflow import log_metric, log_param, log_params, start_run

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_epochs=15
batch_size = 100
loss_criterion = nn.CrossEntropyLoss()
learning_rate = 0.005

def evaluate_model(model, dataset_loader):
    with torch.no_grad():
        n_correct = 0
        n_samples = len(dataset_loader.dataset)
        for labels, images in dataset_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # max returns (output_value ,index)
            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == labels).sum().item()
            
    acc = n_correct / n_samples
    return acc

with start_run(run_name=" 4 Conv, 1st 2x Dropout, 4 dense"):
    log_params({"Epochs": n_epochs, 
                "Batch Size":batch_size, 
                "loss_criterion": loss_criterion._get_name(),
                'learning_rate':learning_rate})


    train = FashionData('data/fashion-mnist_train.csv')
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    
    test = FashionData('data/fashion-mnist_test.csv')
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=True)
    
    network = Net()
    log_param("Network", network)
    log_param('N parameters', sum(p.numel() for p in network.parameters()))

    optimizer = optim.SGD(network.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)

    for epoch in range(n_epochs):
        for i, (labels, images) in enumerate(train_loader):
            network.zero_grad()
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

    train_acc =  evaluate_model(network, train_loader)
    print(f'Accuracy of the network on the train images: {100*train_acc} %')
    log_metric("Train Accuracy", train_acc)

    acc = evaluate_model(network, test_loader)
    print(f'Accuracy of the network on the test images: {100*acc} %')
    log_metric("Test Accuracy", acc)