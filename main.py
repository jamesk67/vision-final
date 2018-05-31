from utils import argParser
from dataloader import BirdLoader
import matplotlib.pyplot as plt
import numpy as np
import models
import torch
import torchvision
import torchvision.transforms as transforms
import pdb
import os
from PIL import Image

def log(logFile, s):
    print(s)
    logFile.write(s + '\n')

def train(net, dataloader, optimizer, criterion, epoch, device, logFile):

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
            log(logFile, '[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    log(logFile, 'Final Summary:   loss: %.3f' %
          (total_loss / i))

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
        for data in dataTestLoader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    log(logFile, '%s Accuracy of the network: %d %%' % (tag,
        100 * correct / total))

    class_correct = list(0. for i in range(555))
    class_total = list(0. for i in range(555))
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

    #print(class_total)
    #print(class_total[403])
    for i in range(555):
        log(logFile, '%s Accuracy of %5s : %2d %%' % (
            tag, dataloader.classes[i], 100 * class_correct[i] / class_total[i]))

def output(net, outputFile, transforms, device):
    directory = os.fsencode('test\\')
    log(outputFile, "path,class")
    for file in os.listdir(directory):
        image = Image.open('test\\' + file.decode())
        image = transforms(image)
        image.unsqueeze_(0)
        image = image.to(device)
        output = net(image)
        _, predicted = torch.max(output.data, 1)
        #print(str(predicted[0].item()))
        log(outputFile, 'test/' + file.decode() + ',' + str(predicted[0].item()))

def main():

    args = argParser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)
    transform = transforms.Compose(
            [
             # TODO: Use these data augmentations later
             transforms.RandomHorizontalFlip(),
             transforms.Resize((400, 400)),
             transforms.RandomCrop((244, 244)),
             transforms.ToTensor(),
             #transforms.ColorJitter(),
             transforms.Normalize((mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
             ])
    #print(os.path.isdir('newtrain'))
    #trainset = torchvision.datasets.ImageFolder('newtrain', transform=transform)

    #cifarLoader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    cifarLoader = BirdLoader(args) # increase batch size to 32
    print('BirdLoader initialized')
    outputFile = open(args.outputfile, 'w+')
    logFile = open(args.logfile, 'w+')
    net = args.model()
    net = net.to(device)
    #print('The log is recorded in ')
    #print(net.logFile.name)

    criterion = net.criterion().to(device)
    optimizer = net.optimizer()

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        net.adjust_learning_rate(optimizer, epoch, args)
        train(net, cifarLoader.trainloader, optimizer, criterion, epoch, device, logFile)
        #if epoch % 1 == 0: # Comment out this part if you want a faster training
            #test(net, cifarLoader, device, logFile, 'Train')
          #  test(net, cifarLoader, device, 'Test')

    output(net, outputFile, transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), device)
    #test(net)
    print("done testing")
    #net.save_state_dict('mytraining.pt')
    torch.save(net.state_dict(), args.modelfile)
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
