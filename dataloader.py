import pandas as pd
import numpy as np
import array
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset  # For custom datasets

class BirdLoader(object):
	"""docstring for BirdLoader"""
	def __init__(self, args):
		super(BirdLoader, self).__init__()
		transform = transforms.Compose(
		    [
		     # TODO: Add data augmentations here
		     transforms.ToTensor(),
		     #transforms.ColorJitter(),
		     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		     ])

		transform_test = transforms.Compose([
		    transforms.ToTensor(),
		    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
		])
        #trainset = CustomDatasetFromImages(csv_path='data/train/labels.csv',
        #								   root_dir='data/train')
        trainset = torchvision.datasets.ImageFolder('./newtrain', transform=transforms)
		#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
		 #                                       download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

        #testset = CustomDatasetFromImages(csv_path='data/test/samples.csv',
        	        #                      root_dir='data/test')
        #testset = torchvision.datasets.ImageFolder('path/to/imagenet')

		#testset = torchvision.datasets.CIFAR10(root='./data', train=False,
		 #                                      download=True, transform=transform_test) 
		#self.testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize,
		 #                                        shuffle=False, num_workers=2)

        #classfile = open('names.txt', 'r')
        self.classes = array('i',(i for i in range(0,555)))
        #classfile.close()
		#self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
		
