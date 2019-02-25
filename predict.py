import json
from collections import OrderedDict

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# loading category number to its name
class_labels = None
with open('cat_to_name.json', 'r') as f:
	class_labels = json.load(f)


def imshow(image, ax=None, title=None):
	"""Imshow for Tensor."""

	if ax is None:
		fig, ax = plt.subplots()

	# PyTorch tensors assume the color channel is the first dimension
	# but matplotlib assumes is the third dimension
	image = image.numpy().transpose((1, 2, 0))

	# Undo preprocessing
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	image = std * image + mean

	# Image needs to be clipped between 0 and 1 or it looks like noise when displayed
	image = np.clip(image, 0, 1)

	# showing image in plot
	ax.imshow(image)

	# setting image title
	if ax and title:
		ax.set_title(title)

	return ax

def process_image(image):
	''' Scales, crops, and normalizes a PIL image for a PyTorch model,
		returns an Numpy array
	'''
	
	# TODO: Process a PIL image for use in a PyTorch model
	image = transforms.functional.resize(image, 256)
	image = transforms.functional.center_crop(image, 224)
	image_tensor = transforms.functional.to_tensor(image)
	image_tensor = transforms.functional.normalize(
		image_tensor, 
		(0.485, 0.456, 0.406), 
		(0.229, 0.224, 0.225)
	)
	return image_tensor.numpy()


def load_model(checkpoint_path='classifier.pth'):
	"""
	loads the checkpoint and rebuilds the model
	"""

	checkpoint = torch.load(checkpoint_path)
	
	if checkpoint['arch'] == 'vgg16':
		model = models.vgg16(pretrained=True)
	else:
		raise Exception("This model architecture was not defined.")

	# Freeze our features parameters
	for param in model.parameters():
		param.requires_grad = False

	# defining custom classifier
	classifier = nn.Sequential(OrderedDict([
		('0', nn.Linear(25088, checkpoint['hidden_size'])),
		('1', nn.ReLU()),
		('2', nn.Dropout()),
		('3', nn.Linear(checkpoint['hidden_size'], checkpoint['output_size'])),
		('4', nn.LogSoftmax(dim=1))
	]))
	model.classifier = classifier
	
	model.class_to_idx = checkpoint['class_to_idx']

	model.load_state_dict(checkpoint['state_dict'])
	
	return model


def predict(image_path, model, topk=5):
	''' Predict the class (or classes) of an image using a trained deep learning model.
	'''

	image = Image.open(image_path)
	normalized_image = torch.Tensor(process_image(image))

	with torch.no_grad():
		torch_image = torch.Tensor(normalized_image)

		shape = [1] + list(torch_image.shape)
		torch_image = torch_image.view(*shape)
		output = model(torch_image)

	probs, classes = output.topk(topk)
	probs = torch.exp(probs)
	idx_to_class = {val: key for key, val in model.class_to_idx.items()}
	top_labels = [idx_to_class[int(c)] for c in classes[0]]
	return probs[0], top_labels


def plot_solution(image_path, probs, classes):
	# Set up plot
	plt.figure(figsize=(6,10))
	ax = plt.subplot(2,1,1)

	# Plot flower
	image = Image.open(image_path)
	normalized_image = torch.Tensor(process_image(image))
	imshow(normalized_image, ax, title='');

	# getting flowers names 
	flowers = [ class_labels[str(number)] for number in classes ]

	# Plot bar chart
	ax = plt.subplot(2,1,2)
	height = [ float(prob) for prob in probs ]
	print(height)
	print(flowers)
	bars = ('A', 'B', 'C', 'D', 'E')
	y_pos = np.arange(len(bars))

	# Create horizontal bars
	ax.barh(y_pos, height, align='center')

	# Create names on the y-axis
	# ax.yticks(y_pos, flowers)
	ax.set_yticks(y_pos)
	ax.set_yticklabels(flowers)
	ax.invert_yaxis()

	plt.show()


# classifier parameters
input_units = 50176
hidden_units=1000
output_units=102
learning_rate=0.001
momentum=0.9
training_epochs=30

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
	print('Using GPU for training model')
else:
	print('Using CPU for training model')


model = load_model()
# image_path = 'flower_data/train/1/image_06734.jpg'
# image_path_alt = 'flower_data/train/1/image_06741.jpg'
# probs, classes = predict(image_path, model, 5)
# print(probs)
# print(classes)
# plot_solution(image_path, probs, classes)