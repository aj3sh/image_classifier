import os
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
from torch.optim import lr_scheduler


# pre parameters
image_size = 224
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


# loading data
data_dir = 'flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

data_transforms = {
	'train': transforms.Compose([
		transforms.RandomRotation(30),
		transforms.RandomResizedCrop(image_size),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(norm_mean, norm_std),
	]),

	'valid': transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(image_size),
		transforms.ToTensor(),
		transforms.Normalize(norm_mean, norm_std),
	]),
}

image_datasets = {
	'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
	'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
}

dataloaders = {
	'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=50, shuffle=True),
	'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=50, shuffle=True)
}


def get_model(input_units, hidden_units, output_units):
	"""
	returns pretrained model
	"""
	model = models.vgg16(pretrained=True)

	# Freeze our features parameters
	for param in model.parameters():
		param.requires_grad = False

	# defining custom classifier
	classifier = nn.Sequential(OrderedDict([
		('0', nn.Linear(25088, hidden_units)),
		('1', nn.ReLU()),
		('2', nn.Dropout()),
		('3', nn.Linear(hidden_units, output_units)),
		('4', nn.LogSoftmax(dim=1))
	]))
	model.classifier = classifier

	return model


def save_model(model):
	"""
	saves the checkpoint of the model
	"""
	model.cpu()
	model_states = {
		'arch': 'vgg16',
		'input_size': input_units,
		'output_size': output_units,
		'hidden_size': hidden_units,
		'state_dict': model.state_dict(), 
		'class_to_idx': model.class_to_idx
	}
	torch.save(model_states, 'classifier.pth')

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

def get_or_load_model():
	"""
	creates news or loads model and return
	"""
	model = get_model(input_units, hidden_units, output_units)
	model.class_to_idx = image_datasets['train'].class_to_idx
	if os.path.exists('classifier.pth'):
		print('Loading saved model')
		model = load_model()
	return model

def save_training_data(optimizer, scheduler, epoch=0):
	"""
	saves the optimizer state
	"""
	optimizer_states = {
		'optimizer_state_dict': optimizer.state_dict(),
		'scheduler_state_dict': scheduler.state_dict(),
		'epoch': epoch,
	}
	torch.save(optimizer_states, 'training_data.pth')

def load_training_data(optimizer, scheduler,checkpoint_path='training_data.pth'):
	"""
	loads the checkpoint and rebuilds the optimizer
	"""
	checkpoint = torch.load(checkpoint_path)
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

	if train_on_gpu:
		for state in optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.cuda()

	return optimizer, scheduler, checkpoint['epoch']

def get_or_load_training_data():
	"""
	creates news or loads optimizer and scheduler and return
	"""
	optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.1)
	epoch = 0
	if os.path.exists('training_data.pth'):
		print('Loading saved training data')
		optimizer, scheduler, epoch = load_training_data(optimizer, scheduler)
	return optimizer, scheduler, epoch


def train(model, criterion, optimizer, scheduler, epochs=30):
	""" trains the model """
	print('Training started')
	print('Number of epochs:', epochs)
	print('Total training datasets:', len(dataloaders['train'].dataset))
	print('Total training batch:', len(dataloaders['train']))

	valid_loss_min = np.Inf

	for epoch in range(1, epochs+1):
		train_loss = 0.0
		valid_loss = 0.0
		accuracy = 0.0

		scheduler.step()
		model.train()
		batch=0
		for data, target in dataloaders['train']:
			if train_on_gpu:
				data = data.cuda()
				target = target.cuda()
				model = model.cuda()

			batch = batch + 1
			# print('Training epoch', epoch, 'of batch', batch)
			optimizer.zero_grad()
			output = model(data)
			
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()*data.size(0)

		# print('Train complete of epoch', epoch)
		
		batch=0
		model.eval()
		for data, target in dataloaders['valid']:
			if train_on_gpu:
				data = data.cuda()
				target = target.cuda()
				model = model.cuda()

			batch = batch + 1
			# print('Validating epoch', epoch, 'of batch', batch)
			output = model(data)
			loss = criterion(output, target)
			valid_loss += loss.item()*data.size(0)
			ps = torch.exp(output)
			top_p, top_class = ps.topk(1, dim=1)
			equals = top_class == target.view(*top_class.shape)
			accuracy += torch.mean(equals.type(torch.FloatTensor))
		
		# calculate average losses
		train_loss = train_loss/len(dataloaders['train'].dataset)
		valid_loss = valid_loss/len(dataloaders['valid'].dataset)
		accuracy = accuracy/len(dataloaders['valid'])

		# print training/validation statistics 
		print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tAccuracy: {:.6f}'.format(
			epoch, train_loss, valid_loss, accuracy))
		
		# save model if validation loss has decreased
		if valid_loss <= valid_loss_min:
			print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
					valid_loss_min,
					valid_loss
				)
			)
			valid_loss_min = valid_loss

			# saving models & training data
			save_model(model)
			save_training_data(optimizer, scheduler, epoch)
	return model


def check_accuracy(model, criterion):
	""" checks accuracy of model """
	model.eval()
	accuracy = 0.0
	i = 0
	for data, target in dataloaders['valid']:
		if train_on_gpu:
			data = data.cuda()
			target = target.cuda()
			model = model.cuda()
			
		i = i + 1
		output = model(data)
		loss = criterion(output, target)
		ps = torch.exp(output)
		top_p, top_class = ps.topk(1, dim=1)
		equals = top_class == target.view(*top_class.shape)
		temp_accuracy = torch.mean(equals.type(torch.FloatTensor))
		accuracy += torch.sum(equals.type(torch.FloatTensor))
		print('Accuracy of batch', i, ':', temp_accuracy)
	accuracy = accuracy/len(dataloaders['valid'].dataset)
	print('Accuracy:', accuracy)
	return accuracy


# classifier parameters
input_units = 50176
hidden_units=4096
output_units=102
learning_rate=0.001
momentum=0.9
training_epochs=50
lr_step_size=5


# checking if GPU is available
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
	print('Using GPU for training model')
else:
	print('Using CPU for training model')


# traning model from here
model = get_or_load_model()
criterion = nn.CrossEntropyLoss()
optimizer, scheduler, epoch = get_or_load_training_data()
trained_model = train(model, criterion, optimizer, scheduler, training_epochs)
check_accuracy(model, criterion)