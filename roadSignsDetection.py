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

labels = [
"Airplanes",
'Ascending Hill',
'Bicycle Parking Lot',
'Bicycles',
'Bicycles And Pedestrians Only',
'Bicycles Crossing',
'Bicycles Only',
'Bumpy Road',
'Bus Only Lane',
'Children Crossing',
'Children Crossing Instruct',
'Crossroad',
'Crosswalk',
'Crosswind',
'Descending Hill',
'Detour',
'Double Bend Left',
'Double Bend Right',
'End Of Dual Carriageway',
'End Of Sign Zone',
'Falling Rocks',
'Follow Directions',
'Height Limit',
'HOV Lane',
'Keep Right',
'Left Curve',
'Left Intersection',
'Left Turn Only',
'Left Lane Decrease',
'Maximum Speed Limit',
'Maximum Weight Limit',
'Maximum Width Limit',
'Merge Left',
'Merge Right',
'Minimum Safe Distance',
'Minimum Speed Limit',
'Motor Vehicles Only',
'Narrow Carriageway',
'No Bicycles',
'No_Buses',
'No Entry',
'No Left Turn',
'No Motor Vehicles',
'No Motor Vehicles And Motorcycles',
'No Motorcycles',
'No Overtaking',
'No Parking',
'No Pedestrians',
'No Right Turn',
'No Stopping',
'No Straight',
'No Tractors',
'No U Turn',
'No Vehicles Carrying Dangerous Substances',
'One Way Left',
'One Way Priority',
'One Way Right',
'One Way Straight',
'Other Dangers',
'Parking Lot',
'Pass Left',
'Pass Left Or Right',
'Pass Right',
'Pedestrian Crossing',
'Pedestrian Only',
'Railway Crossing',
'Right And Left Only',
'Right Curve',
'Right Intersection',
'Right Turn Only',
'Right Lane Decrease',
'Riverside Road',
'Road Closed',
'Roadworks',
'Roundabout',
'Roundabout Instruct',
'Safe Speed',
'Senior Citizens',
'Slippery Road',
'Slow',
'Speed Bumps',
'Start Of Dual Carriageway',
'Stop',
'Straight And Left Only',
'Straight And Right Only',
'Straight Only',
'Traffic Lights',
'T Shape Junction',
'Tunnel',
'Two Way',
'U Turn',
'Wild Animals',
'Yield',
'Y Shape Turn' 
]
def loadModel():
	model = ModelWrapper()
	result = model.loadModel()
	return model
	
def parsePredictions(predictions, threshold):
	# Old version made to return only the highest preditcitons result
	# maxValueImage = [None] * 3
	# maxValueImage[2] = 0
	# for i, pred in enumerate(predictions):
	# 	max = 0.0
	# 	label = 0
	# 	for n, j in enumerate(pred):
	# 		if float(j) > threshold:
	# 			max = float(j)
	# 			label = n
	# 	if(max > maxValueImage[2]):
	# 		maxValueImage[0] = i
	# 		maxValueImage[1] = label
	# 		maxValueImage[2] = max
	# return predictions[maxValueImage[0]], maxValueImage
	output = []
	for i, prediction in enumerate(predictions):
		max = 0.0
		label = 0
		for j, predValue in enumerate(prediction):
			if float(predValue) > max:
				max = float(predValue)
				label = j
		if max > threshold:
			output.append({"arrayId": i, "labelId": label, "predictionValue": max,})

	return output


def partition(input, low, high):
	i = low - 1
	pivot = input[high]

	for j in range(low, high):
		if input[j]["predictionValue"] < pivot["predictionValue"]:
			i = i + 1
			input[i], input[j] = input[j], input[i]

	input[i + 1], input[high] = input[high], input[i + 1]
	return(i + 1)
def quickSort(input, low, high):
	if len(input) == 1:
		return input
	if low < high:
		pi = partition(input, low, high)

		quickSort(input, low, pi - 1)
		quickSort(input, pi + 1, high)
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
		

		images = np.asarray(images)
		predictions = model.predictModel(images)
		parsedPreds = parsePredictions(predictions, 0.8)

		quickSort(parsedPreds, 0, len(parsedPreds) - 1)

		print(parsedPreds)

		image = ImageTk.PhotoImage(image)
		processed = Image.fromarray(bBoxImage)
		processed = ImageTk.PhotoImage(processed)

		if panelA is None or panelB is None:
			panelA = Label(image = image)
			panelA.image = image
			panelA.pack(side="left", padx = 10, pady = 10)

			panelB = Label(image = processed)
			panelB.image = image
			panelB.pack(side="right", padx = 10, pady =10)
		for obj in parsedPreds:
			image = Image.fromarray(images[obj["arrayId"]])
			image = ImageTk.PhotoImage(image)
			panel = Label(image = image)
			panel.image = image
			panel.pack(side="top", padx = 10, pady = 10)
			text = Label(text = labels[obj["labelId"]])
			text.pack(side="top")


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