# from http.server import BaseHTTPRequestHandler, HTTPServer
import http.server
import socketserver
import time
import tensorflow as tf
import os
import sys
import numpy as np
import cv2

from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory

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

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, static_folder=UPLOAD_FOLDER, static_url_path = "/uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = loadModel()

def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			return redirect(url_for('uploaded_file',
									filename=filename))
	return '''
		<!doctype html>
	<title>Upload new File</title>
	<h1>Upload new File</h1>
	<form method=post enctype=multipart/form-data>
	  <input type=file value='Choose file' name=file>
	  <input type=submit value=Upload>
	</form>
	'''
@app.route('/uploads/<filename>/result')
def uploaded_file(filename):
	print(filename)
	print(url_for('static', filename = filename))
	image = cv2.imread(url_for('static', filename = filename)[1:])
	images,bBoxImage = detectShapes(image)
	cv2.imwrite("uploads/" + filename[:-4] + "processed.png", bBoxImage)
	images = np.asarray(images)
	predictions = model.predictModel(images)
	parsedPreds = parsePredictions(predictions, 0.8)
	quickSort(parsedPreds, 0, len(parsedPreds) - 1)
	print(parsedPreds)
	imagesFileNames = []
	for i, img in enumerate(parsedPreds):
		cv2.imwrite("uploads/" + filename[:-4] + "pred" + str(i) + ".png", images[img["arrayId"]])
		imagesFileNames.append(filename[:-4] + "pred" + str(i) + ".png")
		with open("uploads/" + filename[:-4] + "pred" + str(i) + ".txt", "w") as file:
			file.write(str(img["arrayId"]))
			file.write(" ")
			file.write(str(img['labelId']))
			file.write(" ")
			file.write(str(img['predictionValue']))
	print(imagesFileNames)
	return render_template('template.html', my_string="Wheeeee!", my_list=[0,1,2,3,4,5],
		filename = url_for('static', filename = filename), processed = url_for('static', filename = filename[:-4] + "processed.png"),
		images = imagesFileNames, labels = labels, parsedPreds = parsedPreds)	