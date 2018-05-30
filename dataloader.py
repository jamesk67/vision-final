import pandas as pd
import numpy as np
import array
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset  # For custom datasets

class BirdDataset(Dataset):
    def __init__(self, csv_path, transforms):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transforms
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])

        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)

        # Transform image to tensor
        img_as_tensor = self.transforms(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, int(single_image_label))

    def __len__(self):
        return self.data_len

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

		trainset = CustomDatasetFromImages('labels.csv', transform)
        #								   root_dir='data/train')
        #trainset = torchvision.datasets.ImageFolder('./newtrain', transform=transforms)
		#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
		 #                                       download=True, transform=transform)
		self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

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
		
