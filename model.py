import tensorflow as tf 
from dataset import rSigns
from detect_shapes import detectShapes as detectShp
import cv2
import numpy as np
import os
import random

MARKDOWN = "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
class ModelWrapper:
	
	CHECKPOINT_DIR = "./Model"

	train_images, train_labels, model = [0, 0, 0]
	test_images = []
	test_labels = []

	def __init__(self, train_images, train_labels):
		self.model = tf.keras.Sequential()
		self.train_images, self.train_labels = train_images, train_labels
		mask = np.ones(len(self.train_images), dtype = bool)
		for i in range(0, int(len(self.train_images) / 200)):
			for j in range(0, 40):
				temp = int(random.random() * 200) + i * 200

				self.test_images.append(self.train_images[temp])
				self.test_labels.append(self.train_labels[temp])
				mask[temp] = False
		self.train_images = self.train_images[mask, ...]
		self.train_labels = self.train_labels[mask, ...]






	def addLayer(self, layer):
		self.model.add(layer)
		print("Layer ", layer, "was succesfully added")
		print(MARKDOWN)

	def trainModel(self, optimizer, loss, metrics, callbacks, trainingEpochs):
		self.model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
		print("Model compiled")
		print(MARKDOWN)
		with tf.device('/GPU:0'):
			history = self.model.fit(self.train_images, self.train_labels,shuffle = True, callbacks = callbacks, epochs = trainingEpochs)
			print("Model trained")
		print(MARKDOWN)
		results = self.model.evaluate(self.train_images, self.train_labels, verbose = 1)
		print(MARKDOWN)
		return history, results

	def predictModel(self, input):
		return self.model.predict(input) 

	def loadModel(self):
		checkpoints = [self.CHECKPOINT_DIR + "/" + name for name in os.listdir(self.CHECKPOINT_DIR)]
		if checkpoints:
			latest_checkpoint = max(checkpoints, key=os.path.getctime)
			print(MARKDOWN)
			print("Loading model from disk: ", latest_checkpoint)
			print(MARKDOWN)
			self.model =  tf.keras.models.load_model(latest_checkpoint)
			return 0
		else:
			print("Failed to read from disk")
			return 1

# dataset = rSigns()
# train_images, train_labels = dataset.load_data()
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
# callbacks = [
# 		tf.keras.callbacks.EarlyStopping(monitor = "loss", min_delta = 1e-2, patience = 5, verbose = 1),
# 		tf.keras.callbacks.ModelCheckpoint(filepath = "./Model/Model{epoch}", save_best_only = True, monitor = "loss", verbose = 1)
# 		]
# model = ModelWrapper(train_images,train_labels)
# model.addLayer(tf.keras.layers.Conv2D(32, 1,data_format ="channels_last" , padding = "same",input_shape = (32, 32, 3)))
# model.addLayer(tf.keras.layers.Conv2D(32, 3, padding = "same"))
# model.addLayer(tf.keras.layers.Conv2D(32, 1, padding = "same"))
# model.addLayer(tf.keras.layers.MaxPooling2D(2, 2, padding = "valid"))
# model.addLayer(tf.keras.layers.Conv2D(32, 3, 1, padding = "valid"))
# model.addLayer(tf.keras.layers.Conv2D(64, 3, 1, padding = "valid"))
# model.addLayer(tf.keras.layers.Conv2D(128, 3, 1, padding = "valid"))
# model.addLayer(tf.keras.layers.MaxPooling2D(2, 2, padding = "valid"))
# model.addLayer(tf.keras.layers.Flatten())

# model.addLayer(tf.keras.layers.Dropout(0.4))
# model.addLayer(tf.keras.layers.Dense(1024))
# model.addLayer(tf.keras.layers.Dense(256))
# model.addLayer(tf.keras.layers.Dropout(0.2))
# model.addLayer(tf.keras.layers.Dense(96, activation = "softmax"))
# history, result = model.trainModel(optimizer = opt,loss = loss_fn,metrics = ['accuracy'], callbacks = callbacks, trainingEpochs = 100)

# predict = []
# for root, dirs, files in os.walk("Dataset/Source_Materials/Testing", topdown = True):
# 	for name in files:
# 		image = cv2.imread(os.path.join("Dataset/Source_Materials/Testing", name))
# 		predict.append(image)
# predict = np.array(predict)
# predictions = model.predictModel(predict)

# for i, pred in enumerate(predictions):
# 	max = 0.0
# 	label = 0
# 	for n, j in enumerate(pred):
# 		if float(j) > max:
# 			max = float(j)
# 			label = n
# 	print("Label ", label, ": ", max)
# 	print(MARKDOWN)

