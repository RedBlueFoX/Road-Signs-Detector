import numpy as np
import random
import os
import cv2

DATASET_PATH = "Dataset/Source_Materials/Sorted/"
LABELS_PATH = "./labels.txt"



class rSigns:
	images = []
	dirs = []
	labels = []
	def __init__(self, dataset_path = DATASET_PATH, labels_path = LABELS_PATH):
		self._getDirs()
		self._getLabels()
		self._getImages()

	def _getDirs(self, dataset_path = DATASET_PATH):
		for root, dirs, files in os.walk(dataset_path, topdown=False):
			for name in dirs:
				dir = os.path.join(root, name)
				self.dirs.append(dir)
				
	def _getLabels(self,labels_path = LABELS_PATH):
		labels_file = open(labels_path)
		for l in labels_file:
			self.labels.append(l[:-1])

	def _getImages(self):
		for i, d in enumerate(self.dirs):
			for root, dirs, files in os.walk(self.dirs[i], topdown = False):
				for name in files:
					self.images.append(name)

	def _prepare_load_data(self):
		for i, d in enumerate(self.dirs):
			class_dir = []
			for root, dirs, files in os.walk(self.dirs[i], topdown = False):
				
				for name in files:
					image = cv2.imread(os.path.join(d, name))
					class_dir.append(image)
					print("Reading image: ", os.path.join(d, name))
			yield class_dir
	def load_data(self):
		output_images = []
		temp = []
		for i in self._prepare_load_data():
			temp.append(i)
		output_labels = []
		label_counter = 0
		for i, dirObj in enumerate(temp):
			for img in dirObj:

				output_images.append(img)
				output_labels.append(label_counter)
			label_counter = label_counter + 1;	
		
		return np.array(output_images[0:15360]), np.array(output_labels[0:15360]), np.array(output_images[15360:]), np.array(output_labels[15360:]) # 80% of the initial dataset is train data, 20% is test data


def debug():
	dataset = rSigns(sorted = True)

	xtrain, ytrain, xtest, ytest = dataset.load_data()

	print("Size of train data is: ", len(xtrain))
	print("Size of train labels is: ", len(ytrain))
	print("Shape of train data is: ", xtrain.shape)
	print("Shape of train labels is: ", ytrain.shape)
	print("Size of test data is: ", len(xtest))
	print("Size of test labels is: ", len(ytest))
	print("Shape of test data is: ", xtest.shape)
	print("Shape of test labels is: ", ytest.shape)

