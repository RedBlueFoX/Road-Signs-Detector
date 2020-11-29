import tensorflow as tf 
from dataset import rSigns
from detect_shapes import detectShapes as detectShp
import cv2
import numpy as np
import os

MARKDOWN = "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
class ModelWrapper:
	dataset = rSigns()
	CHECKPOINT_DIR = "./Model"

	train_images, train_labels, test_images, test_labels, model = [0, 0, 0, 0, 0]

	def __init__(self):
		self.model = tf.keras.Sequential()
		self.train_images, self.train_labels, self.test_images, self.test_labels = self.dataset.load_data()
		print(self.train_images.shape)
		print(self.train_labels.shape)

	def addLayer(self, layer):
		self.model.add(layer)
		print("Layer ", layer, "was succesfully added")
		print(MARKDOWN)

	def trainModel(self, optimizer, loss, metrics, trainingEpochs):
		self.model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
		print("Model compiled")
		print(MARKDOWN)
		print(self.train_images)
		print(self.train_labels.shape)
		self.model.fit(self.train_images, self.train_labels,shuffle = True,  epochs = trainingEpochs)
	
		print("Model trained")
		print(MARKDOWN)
		results = self.model.evaluate(self.test_images, self.test_labels, verbose = 2)
		return results

	def predictModel(self, input):
		return self.model.predict(input) 

	def loadModel(self):
		checkpoints = [self.CHECKPOINT_DIR + "/" + name for name in os.listdir(self.CHECKPOINT_DIR)]
		if checkpoints:
			latest_checkpoint = max(checkpoints, key=os.path.getctime)
			print(MARKDOWN)
			print("Loading model from disk: ", latest_checkpoint)
			print(MARKDOWN)
			return tf.keras.models.load_model(latest_checkpoint)
		else:
			print("Failed to read from disk")
			return 0

model = ModelWrapper()
model.addLayer(tf.keras.layers.Input(shape = (32,32,3)))
model.addLayer(tf.keras.layers.Flatten())
model.addLayer(tf.keras.layers.Dense(3072))
model.addLayer(tf.keras.layers.Dense(1024))
model.addLayer(tf.keras.layers.Dense(512))
model.addLayer(tf.keras.layers.Dense(256))
model.addLayer(tf.keras.layers.Dense(128))
model.addLayer(tf.keras.layers.Dropout(0.4))
model.addLayer(tf.keras.layers.Dense(96))
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.trainModel(optimizer = 'adam',loss = loss_fn,metrics = ['accuracy'], trainingEpochs = 10)