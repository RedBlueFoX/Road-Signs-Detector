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

	def __init__(self, dataset):
		self.model = tf.keras.Sequential()
		self.train_images, self.train_labels = dataset.load_data()
		
		for i in range(0, int(len(self.train_images) / 200)):
			for j in range(0, 40):
				temp = int(random.random() * 200) + i * 200
				print(j, " | ", i , " | ", temp)
				print(self.train_images.shape)
				self.test_images.append(self.train_images[temp])
				self.test_labels.append(self.train_labels[temp])
				np.delete(self.train_images, temp)
				np.delete(self.train_labels, temp)
		self.test_images = np.array(self.test_images)
		list(filter(None, self.train_images))
		list(filter(None, self.train_labels))
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
		with tf.device('/GPU:0'):
			history = self.model.fit(self.train_images, self.train_labels,shuffle = True,  epochs = trainingEpochs)
			print("Model trained")
		print(MARKDOWN)
		results = self.model.evaluate(self.train_images, self.train_labels, verbose = 2)
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
			return tf.keras.models.load_model(latest_checkpoint)
		else:
			print("Failed to read from disk")
			return 0

dataset = rSigns()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
result = [None] * 15
model = ModelWrapper(dataset)
model.addLayer(tf.keras.layers.Conv2D(64, 3, activation = "relu",data_format ="channels_last" ,input_shape = (32, 32, 3)))
model.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
model.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu"))
model.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
model.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu"))
model.addLayer(tf.keras.layers.Flatten())
model.addLayer(tf.keras.layers.Dense(512))
model.addLayer(tf.keras.layers.Dense(256))
model.addLayer(tf.keras.layers.Dropout(0.4))
model.addLayer(tf.keras.layers.Dense(96, activation = "softmax"))
history, result[0] = model.trainModel(optimizer = 'adam',loss = loss_fn,metrics = ['accuracy'], trainingEpochs = 20)

model1 = ModelWrapper(dataset)
model1.addLayer(tf.keras.layers.Conv2D(1024, 3, activation = "relu",data_format ="channels_last" ,input_shape = (32, 32, 3)))
model1.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
model1.addLayer(tf.keras.layers.Conv2D(256, 3, activation = "relu"))
model1.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
model1.addLayer(tf.keras.layers.Conv2D(64, 3, activation = "relu"))
model1.addLayer(tf.keras.layers.Flatten())
model1.addLayer(tf.keras.layers.Dense(512))
model1.addLayer(tf.keras.layers.Dense(256))
model1.addLayer(tf.keras.layers.Dropout(0.4))
model1.addLayer(tf.keras.layers.Dense(96, activation = "softmax"))
history1, result[1] = model1.trainModel(optimizer = 'adam',loss = loss_fn,metrics = ['accuracy'], trainingEpochs = 20)

# model2 = ModelWrapper(dataset)
# model2.addLayer(tf.keras.layers.Conv2D(64, 2, activation = "relu",data_format ="channels_last" ,input_shape = (32, 32, 3)))
# model2.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model2.addLayer(tf.keras.layers.Conv2D(32, 2, activation = "relu"))
# model2.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model2.addLayer(tf.keras.layers.Conv2D(32, 2, activation = "relu"))
# model2.addLayer(tf.keras.layers.Flatten())
# model2.addLayer(tf.keras.layers.Dense(512))
# model2.addLayer(tf.keras.layers.Dense(256))
# model2.addLayer(tf.keras.layers.Dropout(0.4))
# model2.addLayer(tf.keras.layers.Dense(96, activation = "relu"))
# history2, result[2] = model2.trainModel(optimizer = 'adam',loss = loss_fn,metrics = ['accuracy'], trainingEpochs = 20)

# model3 = ModelWrapper(dataset)
# model3.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu",data_format ="channels_last" ,input_shape = (32, 32, 3)))
# model3.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model3.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu"))
# model3.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model3.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu"))
# model3.addLayer(tf.keras.layers.Flatten())
# model3.addLayer(tf.keras.layers.Dense(512))
# model3.addLayer(tf.keras.layers.Dense(256))
# model3.addLayer(tf.keras.layers.Dropout(0.4))
# model3.addLayer(tf.keras.layers.Dense(96, activation = "relu"))
# history3, result[3] = model3.trainModel(optimizer = 'adam',loss = loss_fn,metrics = ['accuracy'], trainingEpochs = 20)

# model4 = ModelWrapper(dataset)
# model4.addLayer(tf.keras.layers.Conv2D(64, 3, activation = "relu",data_format ="channels_last" ,input_shape = (32, 32, 3)))
# model4.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model4.addLayer(tf.keras.layers.Conv2D(64, 3, activation = "relu"))
# model4.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model4.addLayer(tf.keras.layers.Conv2D(64, 3, activation = "relu"))
# model4.addLayer(tf.keras.layers.Flatten())
# model4.addLayer(tf.keras.layers.Dense(512))
# model4.addLayer(tf.keras.layers.Dense(256))
# model4.addLayer(tf.keras.layers.Dropout(0.4))
# model4.addLayer(tf.keras.layers.Dense(96, activation = "relu"))
# history4, result[4] = model4.trainModel(optimizer = 'adam',loss = loss_fn,metrics = ['accuracy'], trainingEpochs = 20)

# model5 = ModelWrapper(dataset)
# model5.addLayer(tf.keras.layers.Conv2D(64, 3, activation = "relu",data_format ="channels_last" ,input_shape = (32, 32, 3)))
# model5.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model5.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu"))
# model5.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model5.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu"))
# model5.addLayer(tf.keras.layers.Flatten())
# model5.addLayer(tf.keras.layers.Dense(256))
# model5.addLayer(tf.keras.layers.Dense(128))
# model5.addLayer(tf.keras.layers.Dropout(0.4))
# model5.addLayer(tf.keras.layers.Dense(96, activation = "relu"))
# history5, result[5] = model5.trainModel(optimizer = 'adam',loss = loss_fn,metrics = ['accuracy'], trainingEpochs = 20)

# model6 = ModelWrapper(dataset)
# model6.addLayer(tf.keras.layers.Conv2D(64, 3, activation = "relu",data_format ="channels_last" ,input_shape = (32, 32, 3)))
# model6.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model6.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu"))
# model6.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model6.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu"))
# model6.addLayer(tf.keras.layers.Flatten())
# model6.addLayer(tf.keras.layers.Dense(512))
# model6.addLayer(tf.keras.layers.Dense(256))
# model6.addLayer(tf.keras.layers.Dropout(0.2))
# model6.addLayer(tf.keras.layers.Dense(96, activation = "relu"))
# history6, result[6] = model6.trainModel(optimizer = 'adam',loss = loss_fn,metrics = ['accuracy'], trainingEpochs = 20)

# model7 = ModelWrapper(dataset)
# model7.addLayer(tf.keras.layers.Conv2D(64, 3, activation = "selu",data_format ="channels_last" ,input_shape = (32, 32, 3)))
# model7.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model7.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "selu"))
# model7.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model7.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "selu"))
# model7.addLayer(tf.keras.layers.Flatten())
# model7.addLayer(tf.keras.layers.Dense(512))
# model7.addLayer(tf.keras.layers.Dense(256))
# model7.addLayer(tf.keras.layers.Dropout(0.4))
# model7.addLayer(tf.keras.layers.Dense(96, activation = "relu"))
# history7, result[7] = model7.trainModel(optimizer = 'adam',loss = loss_fn,metrics = ['accuracy'], trainingEpochs = 20)

# model8 = ModelWrapper(dataset)
# model8.addLayer(tf.keras.layers.Conv2D(64, 4, activation = "relu",data_format ="channels_last" ,input_shape = (32, 32, 3)))
# model8.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model8.addLayer(tf.keras.layers.Conv2D(64, 4, activation = "relu"))
# model8.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model8.addLayer(tf.keras.layers.Conv2D(64, 4, activation = "relu"))
# model8.addLayer(tf.keras.layers.Flatten())
# model8.addLayer(tf.keras.layers.Dense(512))
# model8.addLayer(tf.keras.layers.Dense(256))
# model8.addLayer(tf.keras.layers.Dropout(0.4))
# model8.addLayer(tf.keras.layers.Dense(96, activation = "relu"))
# history8, result[8] = model8.trainModel(optimizer = 'adam',loss = loss_fn,metrics = ['accuracy'], trainingEpochs = 20)

# model9 = ModelWrapper(dataset)
# model9.addLayer(tf.keras.layers.Conv2D(64, 3, activation = "relu",data_format ="channels_last" ,input_shape = (32, 32, 3)))
# model9.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model9.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu"))
# model9.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model9.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu"))
# model9.addLayer(tf.keras.layers.Flatten())
# model9.addLayer(tf.keras.layers.Dense(512))
# model9.addLayer(tf.keras.layers.Dense(256))
# model9.addLayer(tf.keras.layers.Dense(128))
# model9.addLayer(tf.keras.layers.Dropout(0.4))
# model9.addLayer(tf.keras.layers.Dense(96, activation = "relu"))
# history9, result[9] = model9.trainModel(optimizer = 'adam',loss = loss_fn,metrics = ['accuracy'], trainingEpochs = 20)

# model10 = ModelWrapper(dataset)
# model10.addLayer(tf.keras.layers.Conv2D(64, 3, activation = "relu",data_format ="channels_last" ,input_shape = (32, 32, 3)))
# model10.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model10.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu"))
# model10.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model10.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu"))
# model10.addLayer(tf.keras.layers.Flatten())
# model10.addLayer(tf.keras.layers.Dense(512))
# model10.addLayer(tf.keras.layers.Dense(256))
# model10.addLayer(tf.keras.layers.Dropout(0.4))
# model10.addLayer(tf.keras.layers.Dense(96, activation = "softmax"))
# history, result[10] = model10.trainModel(optimizer = 'adam',loss = loss_fn,metrics = ['accuracy'], trainingEpochs = 20)

# model11 = ModelWrapper(dataset)
# model11.addLayer(tf.keras.layers.Conv2D(64, 3, activation = "relu",data_format ="channels_last" ,input_shape = (32, 32, 3)))
# model11.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model11.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu"))
# model11.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model11.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu"))
# model11.addLayer(tf.keras.layers.Flatten())
# model11.addLayer(tf.keras.layers.Dense(512))
# model11.addLayer(tf.keras.layers.Dense(256))
# model11.addLayer(tf.keras.layers.Dropout(0.6))
# model11.addLayer(tf.keras.layers.Dense(96, activation = "relu"))
# history, result[11] = model11.trainModel(optimizer = 'adam',loss = loss_fn,metrics = ['accuracy'], trainingEpochs = 20)

# model12 = ModelWrapper(dataset)
# model12.addLayer(tf.keras.layers.Conv2D(64, 2, activation = "relu",data_format ="channels_last" ,input_shape = (32, 32, 3)))
# model12.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model12.addLayer(tf.keras.layers.Conv2D(64, 2, activation = "relu"))
# model12.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model12.addLayer(tf.keras.layers.Conv2D(64, 2, activation = "relu"))
# model12.addLayer(tf.keras.layers.Flatten())
# model12.addLayer(tf.keras.layers.Dense(512))
# model12.addLayer(tf.keras.layers.Dense(256))
# model12.addLayer(tf.keras.layers.Dropout(0.4))
# model12.addLayer(tf.keras.layers.Dense(96, activation = "relu"))
# history, result[12] = model12.trainModel(optimizer = 'adam',loss = loss_fn,metrics = ['accuracy'], trainingEpochs = 20)

# model13 = ModelWrapper(dataset)
# model13.addLayer(tf.keras.layers.Conv2D(64, 3, activation = "relu",data_format ="channels_last" ,input_shape = (32, 32, 3)))
# model13.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model13.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "softmax"))
# model13.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model13.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu"))
# model13.addLayer(tf.keras.layers.Flatten())
# model13.addLayer(tf.keras.layers.Dense(512))
# model13.addLayer(tf.keras.layers.Dense(256))
# model13.addLayer(tf.keras.layers.Dropout(0.4))
# model13.addLayer(tf.keras.layers.Dense(96, activation = "relu"))
# history, result[13] = model13.trainModel(optimizer = 'adam',loss = loss_fn,metrics = ['accuracy'], trainingEpochs = 20)

# model14 = ModelWrapper(dataset)
# model14.addLayer(tf.keras.layers.Conv2D(64, 3, activation = "relu",data_format ="channels_last" ,input_shape = (32, 32, 3)))
# model14.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model14.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu"))
# model14.addLayer(tf.keras.layers.MaxPooling2D(2, 2))
# model14.addLayer(tf.keras.layers.Conv2D(32, 3, activation = "relu"))
# model14.addLayer(tf.keras.layers.Flatten())
# model14.addLayer(tf.keras.layers.Dense(512))
# model14.addLayer(tf.keras.layers.Dense(256))
# model14.addLayer(tf.keras.layers.Dropout(0.4))
# model14.addLayer(tf.keras.layers.Dense(96, activation = "relu"))
# history, result[14] = model14.trainModel(optimizer = 'adam',loss = loss_fn,metrics = ['accuracy'], trainingEpochs = 20)



for i, x in enumerate(result):
	print("Result of model №" + str(i) + " is: " + str(x))