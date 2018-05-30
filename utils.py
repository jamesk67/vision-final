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
	parser.add_argument('--lr', default=0.01, type=float)
	parser.add_argument('--batchSize', default=4, type=int)
	parser.add_argument('--epochs', default=1, type=int)
	parser.add_argument('--model', type=model_class)
	parser.add_argument('--logfile', type=str)
	parser.add_argument('--outputfile', type=str)
	parser.add_argument('--modelfile', type=str)
	return parser.parse_args()

