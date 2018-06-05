from utils import argParser
from dataloader import BirdLoader

import matplotlib.pyplot as plt
import numpy as np
import models
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import pdb
import os
from torch.autograd import Variable
from PIL import Image
import datetime

def log(logFile, s):
    print(s)
    logFile.write(s + '\n')

def train(net, dataloader, optimizer, criterion, epoch, device, logFile):

    running_loss = 0.0
    total_loss = 0.0

    for i, (inputs, labels) in enumerate(dataloader, 0):
        # get the inputs
        #inputs, labels = data
        inputs = Variable(inputs).to(device)
        labels = Variable(labels).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs).to(device)

        # regression tensor for MSELoss, comment out the following four lines
        # to use CrossEntropyLoss
        #rlabels = labels.new_zeros((labels.size()[0], 10), dtype=torch.float)
        #for i in range(labels.size()[0]):
         #   rlabels[i, labels[i].item()] = 1
        #labels = rlabels
        #print(outputs.size())
        #print(labels.size())
        loss = criterion(outputs, labels).to(device)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total_loss += loss.item()
        if (i + 1) % 2000 == 0:    # print every 2000 mini-batches
            log(logFile, '[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    log(logFile, 'Final Summary:   loss: %.3f' %
          (total_loss / i))
    return total_loss / i


def test(net, dataloader, device, logFile, tag=''):
    correct = 0
    total = 0
    dataTestLoader = dataloader.trainloader
    '''
    if tag == 'Train':
        dataTestLoader = dataloader.trainloader
    else:
        dataTestLoader = dataloader.testloader
    '''
    with torch.no_grad():
        for (images, labels) in dataTestLoader:
            #images, labels = data
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            outputs = net(images).to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    log(logFile, '%s Accuracy of the network: %d %%' % (tag,
        100 * correct / total))

    class_correct = list(0. for i in range(555))
    class_total = list(0. for i in range(555))
    with torch.no_grad():
        for (images, labels) in dataTestLoader:
            #images, labels = data
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            outputs = net(images).to(device)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    #print(class_total)
    #print(class_total[403])
    for i in range(555):
        log(logFile, '%s Accuracy of %5s : %2d %%' % (
            tag, dataloader.classes[i], 100 * class_correct[i] / class_total[i]))

def output(net, outputFile, transforms, device):
    directory = os.fsencode('test/')
    log(outputFile, "path,class")
    with torch.no_grad():
        for file in os.listdir(directory):
            image = Image.open('test/' + file.decode())
            image = transforms(image).to(device)
            image.unsqueeze_(0)
            output = net(image).to(device)
            _, predicted = torch.max(output.data, 1)
            #print(str(predicted[0].item()))
            log(outputFile, 'test/' + file.decode() + ',' + str(predicted[0].item()))

def main():

    args = argParser()
    #torch.cuda.set_device(1)
    device = torch.device("cuda:0")
    #device = torch.device('cpu')
    print(device)
    transform = transforms.Compose(
            [
             # TODO: Use these data augmentations later
             transforms.RandomHorizontalFlip(),
             transforms.RandomResizedCrop(224),
             transforms.ToTensor(),
             #transforms.ColorJitter(),
             transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
             ])
    #print(os.path.isdir('newtrain'))
    #trainset = torchvision.datasets.ImageFolder('newtrain', transform=transform)

    #cifarLoader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    cifarLoader = BirdLoader(args)
    #print('BirdLoader initialized')
    outputFile = open(args.outputfile, 'w+')
    logFile = open(args.logfile, 'w+')
    net = args.model()
    net = net.to(device)
    #print('The log is recorded in ')
    #print(net.logFile.name)

    
    if not args.getoutput:
        criterion = net.criterion().to(device)
        params = list(net.parameters())

        for p in params:
            print(p.requires_grad)

        #optimizer = optim.Adam(params)
        optimizer = net.optimizer()
        convergeCount = 5
        currentLoss = float("inf")
        startTime = datetime.datetime.now()
        log(logFile, 'Training began at ' + str(startTime))
        for epoch in range(args.epochs):  # loop over the dataset multiple times
            log(logFile, 'Epoch ' + str(epoch + 1))
            net.adjust_learning_rate(optimizer, epoch, args)
            loss = train(net, cifarLoader.trainloader, optimizer, criterion, epoch, device, logFile)
            if epoch % 1 == 0: # Comment out this part if you want a faster training
                test(net, cifarLoader, device, logFile, 'Train')
            if abs(currentLoss - loss) < 0.001:
                convergeCount -= 1
            else:
                convergeCount = 5
            if loss < currentLoss:
                currentLoss = loss
                torch.save(net.state_dict(), args.modelfile)
            if convergeCount == 0:
                break
        endTime = datetime.datetime.now()
        log(logFile, 'Training ended at ' + str(endTime) + '. It trained for ' + str(endTime - startTime))
    else:
        net.load_state_dict(torch.load(args.modelfile))

    output(net, outputFile, transforms.Compose([transforms.Resize(256), 
                            transforms.CenterCrop(224), 
                            transforms.ToTensor(), 
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]), device)
    #test(net)
    
    #net.save_state_dict('mytraining.pt')

    log(logFile, 'The log is recorded in ')
    log(logFile, args.logfile)
    log(logFile, 'The output is recorded in ')
    log(logFile, args.outputfile)
    log(logFile, 'The model is recorded in ')
    log(logFile, args.modelfile)
    logFile.close()
    outputFile.close()

if __name__ == '__main__':
    main()
