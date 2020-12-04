import tensorflow as tf
import os
import sys
import numpy as np
import cv2
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

from dataset import rSigns
from model import ModelWrapper
from detect_shapes import detectShapes 

def loadModel():
	model = ModelWrapper()
	result = model.loadModel()
	return model
	
def parsePredictions(predictions):

	maxValueImage = [None] * 3
	maxValueImage[2] = 0
	for i, pred in enumerate(predictions):
		max = 0.0
		label = 0
		for n, j in enumerate(pred):
			if float(j) > max:
				max = float(j)
				label = n
		if(max > maxValueImage[2]):
			maxValueImage[0] = i
			maxValueImage[1] = label
			maxValueImage[2] = max
	return predictions[maxValueImage[0]], maxValueImage



def select_image():
	global panelA, panelB

	path = filedialog.askopenfilename()
	print(path)
	if len(path) > 0:
		img = cv2.imread(path)
		image = img.copy()
		xImageSize = image.shape[0]
		yImageSize = image.shape[1]
		imageProportions = xImageSize / yImageSize
		image = cv2.resize(image, (540, int(540 * imageProportions)))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		images,bBoxImage = detectShapes(image)
		image = Image.fromarray(image)
		
		print(images.shape)
		images = np.asarray(images)
		predictions = model.predictModel(images)
		predictedImage, info = parsePredictions(predictions)
		image = ImageTk.PhotoImage(image)
		processed = Image.fromarray(predictedImage)
		processed = ImageTk.PhotoImage(processed)
		print(info)
		if panelA is None or panelB is None:
			panelA = Label(image = image)
			panelA.image = image
			panelA.pack(side="left", padx = 10, pady = 10)

			panelB = Label(image=processed)
			panelB.image = processed
			panelB.pack(side="right", padx=10, pady =10)

		else:
			panelA.configure(image = image)
			panelB.configure(image = processed)
			panelA.image = image
			panelB.image = processed

model = loadModel()
root = Tk()
panelA = None
panelB = None

btn = Button(root, text = "FILE", command = select_image, width = 70, height = 30, bg = "blue", fg = "yellow")

btn.pack(side="bottom", fill = "both", expand = "yes")


root.mainloop()