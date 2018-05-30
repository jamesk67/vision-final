from utils import argParser
#from dataloader import BirdLoader
import matplotlib.pyplot as plt
import numpy as np
import models
import torch
import torchvision
import torchvision.transforms as transforms
import pdb

def train(net, dataloader, optimizer, criterion, epoch, device):

    running_loss = 0.0
    total_loss = 0.0

    for i, data in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        # regression tensor for MSELoss, comment out the following four lines
        # to use CrossEntropyLoss
        #rlabels = labels.new_zeros((labels.size()[0], 10), dtype=torch.float)
        #for i in range(labels.size()[0]):
         #   rlabels[i, labels[i].item()] = 1
        #labels = rlabels
        #print(outputs.size())
        #print(labels.size())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total_loss += loss.item()
        if (i + 1) % 2000 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    print('Final Summary:   loss: %.3f' %
          (total_loss / i))

def test(net, dataloader, device, tag=''):
    correct = 0
    total = 0
    dataTestLoader = dataloader
    '''
    if tag == 'Train':
        dataTestLoader = dataloader.trainloader
    else:
        dataTestLoader = dataloader.testloader
    '''
    with torch.no_grad():
        for data in dataTestLoader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('%s Accuracy of the network: %d %%' % (tag,
        100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in dataTestLoader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(555):
      print('%s Accuracy of %5s : %2d %%' % (
       tag, dataloader.classes[i], 100 * class_correct[i] / class_total[i]))

def main():

    args = argParser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    transform = transforms.Compose(
            [
             # TODO: Add data augmentations here
             transforms.Resize((96, 96)),
             transforms.RandomVerticalFlip(),
             transforms.ToTensor(),
             #transforms.ColorJitter(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
    trainset = torchvision.datasets.ImageFolder('./newtrain', transform=transform)

    cifarLoader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    net = args.model()
    net = net.to(device)
    #print('The log is recorded in ')
    #print(net.logFile.name)

    criterion = net.criterion().to(device)
    optimizer = net.optimizer()

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        net.adjust_learning_rate(optimizer, epoch, args)
        train(net, cifarLoader, optimizer, criterion, epoch, device)
        if epoch % 1 == 0: # Comment out this part if you want a faster training
           test(net, cifarLoader, device, 'Train')
          #  test(net, cifarLoader, device, 'Test')


    #test(net)
    print("done testing")
    #net.save_state_dict('mytraining.pt')
    #torch.save(net.state_dict(), './models')
    #print('The log is recorded in ')
    #print(net.logFile.name)

if __name__ == '__main__':
    main()
