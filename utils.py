import matplotlib.pyplot as plt
import numpy as np
import argparse
import models
import torch
import torch.optim as optim

def model_class(class_name):
    return getattr(models, class_name)


def argParser():
	parser = argparse.ArgumentParser(description='PyTorch Vision Final')
	parser.add_argument('--getoutput', default=False, type=bool)
	parser.add_argument('--lr', default=0.01, type=float)
	parser.add_argument('--batchSize', default=256, type=int)
	parser.add_argument('--epochs', default=50, type=int)
	parser.add_argument('--model', type=model_class)
	parser.add_argument('--logfile', default='log.txt', type=str)
	parser.add_argument('--outputfile', default='output.csv', type=str)
	parser.add_argument('--modelfile', default='model.pth', type=str)
	return parser.parse_args()

