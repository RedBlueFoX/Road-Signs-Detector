import cv2
import numpy as np
import os
import random


OUTPUT_DIMENSIONS = (256, 256)
IMAGE_CLASSES = [
    "Dataset/Source_Materials/Sorted/Airplanes",
"Dataset/Source_Materials/Sorted/Ascending Hill",
"Dataset/Source_Materials/Sorted/Bicycle Parking Lot",
"Dataset/Source_Materials/Sorted/Bicycles",
"Dataset/Source_Materials/Sorted/Bicycles and Pedestrians Only",
"Dataset/Source_Materials/Sorted/Bicycles Crossing",
"Dataset/Source_Materials/Sorted/Bicycles Only",
"Dataset/Source_Materials/Sorted/Bumpy Road",
"Dataset/Source_Materials/Sorted/Bus Only Lane",
"Dataset/Source_Materials/Sorted/Children Crossing",
"Dataset/Source_Materials/Sorted/Children Crossing Instruct",
"Dataset/Source_Materials/Sorted/Crossroad",
"Dataset/Source_Materials/Sorted/Crosswalk",
"Dataset/Source_Materials/Sorted/Crosswind",
"Dataset/Source_Materials/Sorted/Descending Hill",
"Dataset/Source_Materials/Sorted/Detour",
"Dataset/Source_Materials/Sorted/Double-Bend Left",
"Dataset/Source_Materials/Sorted/Double-Bend right",
"Dataset/Source_Materials/Sorted/End of Dual Carriageway",
"Dataset/Source_Materials/Sorted/End of Sign Zone",
"Dataset/Source_Materials/Sorted/Falling Rocks",
"Dataset/Source_Materials/Sorted/Follow Directions",
"Dataset/Source_Materials/Sorted/Height limit",
"Dataset/Source_Materials/Sorted/HOV Lane",
"Dataset/Source_Materials/Sorted/Keep Right",
"Dataset/Source_Materials/Sorted/Left Curve",
"Dataset/Source_Materials/Sorted/Left Intersection",
"Dataset/Source_Materials/Sorted/Left Turn Only",
"Dataset/Source_Materials/Sorted/Left-Lane Decrease",
"Dataset/Source_Materials/Sorted/Maximum Speed Limit",
"Dataset/Source_Materials/Sorted/Maximum Weight Limit",
"Dataset/Source_Materials/Sorted/Maximum width Limit",
"Dataset/Source_Materials/Sorted/Merge-Left",
"Dataset/Source_Materials/Sorted/Merge-Right",
"Dataset/Source_Materials/Sorted/Minimum Safe Distance",
"Dataset/Source_Materials/Sorted/Minimum Speed Limit",
"Dataset/Source_Materials/Sorted/Motor Vehicles Only",
"Dataset/Source_Materials/Sorted/Narrow Carriageway",
"Dataset/Source_Materials/Sorted/No Bicycles",
"Dataset/Source_Materials/Sorted/No Buses",
"Dataset/Source_Materials/Sorted/No Entry",
"Dataset/Source_Materials/Sorted/No Left Turn",
"Dataset/Source_Materials/Sorted/No Motor Vehicles",
"Dataset/Source_Materials/Sorted/No Motor Vehicles and Motorcycles",
"Dataset/Source_Materials/Sorted/No Motorcycles",
"Dataset/Source_Materials/Sorted/No Overtaking",
"Dataset/Source_Materials/Sorted/No Parking",
"Dataset/Source_Materials/Sorted/No Pedestrians",
"Dataset/Source_Materials/Sorted/No Right Turn",
"Dataset/Source_Materials/Sorted/No Stopping",
"Dataset/Source_Materials/Sorted/No Straight",
"Dataset/Source_Materials/Sorted/No Tractors",
"Dataset/Source_Materials/Sorted/No Trucks",
"Dataset/Source_Materials/Sorted/No U-Turn",
"Dataset/Source_Materials/Sorted/No Vehicles Carrying Dangerous Substances",
"Dataset/Source_Materials/Sorted/One Way Left",
"Dataset/Source_Materials/Sorted/One Way Priority",
"Dataset/Source_Materials/Sorted/One Way Right",
"Dataset/Source_Materials/Sorted/One Way Straight",
"Dataset/Source_Materials/Sorted/Other Dangers",
"Dataset/Source_Materials/Sorted/Parking Lot",
"Dataset/Source_Materials/Sorted/Pass Left",
"Dataset/Source_Materials/Sorted/Pass Left or Right",
"Dataset/Source_Materials/Sorted/Pass Left or Right Instr",
"Dataset/Source_Materials/Sorted/Pass Right",
"Dataset/Source_Materials/Sorted/Pedestrian Crossing",
"Dataset/Source_Materials/Sorted/Pedestrians Only",
"Dataset/Source_Materials/Sorted/Railway Crossing",
"Dataset/Source_Materials/Sorted/Right and Left Only",
"Dataset/Source_Materials/Sorted/Right Curve",
"Dataset/Source_Materials/Sorted/Right Intersection",
"Dataset/Source_Materials/Sorted/Right Turn Only",
"Dataset/Source_Materials/Sorted/Right-Lane Decrease",
"Dataset/Source_Materials/Sorted/Riverside Road",
"Dataset/Source_Materials/Sorted/Road Closed",
"Dataset/Source_Materials/Sorted/Roadworks",
"Dataset/Source_Materials/Sorted/Roundabout",
"Dataset/Source_Materials/Sorted/Roundabout Instruct",
"Dataset/Source_Materials/Sorted/Safe Speed",
"Dataset/Source_Materials/Sorted/Senior Citizens",
"Dataset/Source_Materials/Sorted/Slippery Road",
"Dataset/Source_Materials/Sorted/Slow",
"Dataset/Source_Materials/Sorted/Speed Bumps",
"Dataset/Source_Materials/Sorted/Start of Dual Carriageway",
"Dataset/Source_Materials/Sorted/Stop",
"Dataset/Source_Materials/Sorted/Straight And Left Only",
"Dataset/Source_Materials/Sorted/Straight And Right Only",
"Dataset/Source_Materials/Sorted/Straight Only",
"Dataset/Source_Materials/Sorted/Traffic Lights",
"Dataset/Source_Materials/Sorted/T-Shape",
"Dataset/Source_Materials/Sorted/Tunnel",
"Dataset/Source_Materials/Sorted/Two-Way",
"Dataset/Source_Materials/Sorted/U-Turn",
"Dataset/Source_Materials/Sorted/Wild Animals",
"Dataset/Source_Materials/Sorted/Yield",
"Dataset/Source_Materials/Sorted/Y-Shape",
] 


def markDir(dir = "Dataset/Source_Materials/Unsorted"):
	onlyfiles = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
	textArr = []
	imgArr = []
	for n in onlyfiles:

		if n[-4:] == ".txt":
			textArr.append(n)
		if n[-4:] == ".jpg":
			imgArr.append(n)

	return textArr, imgArr
def readInfoFromFile(fileName,pathToDataset = "Dataset/Source_Materials/Unsorted/" ):
	f = open(pathToDataset + fileName, "r")
	info = f.readline().split()
	output = []
	while(info):
		output.append(info)
		info = f.readline().split()
	f.close()
	return output

def cropInput(img, info):
	image = img.copy()
	IMAGE_DIMENSIONS = image.shape[:2]
	objPos = [info[1], info[2]]
	objSize = info[3:]
	print("Image Dimensions are:", IMAGE_DIMENSIONS)
	objPosCoordinates = [float(objPos[0]) * IMAGE_DIMENSIONS[1], float(objPos[1]) * IMAGE_DIMENSIONS[0]]
	print("Object position is: ", objPosCoordinates)
	objSizeDenormalized = [float(objSize[0]) * IMAGE_DIMENSIONS[1], float(objSize[1]) * IMAGE_DIMENSIONS[0]]
	print("Object size is: ", objSizeDenormalized)
	objBox = [[objPosCoordinates[0] - objSizeDenormalized[0] / 2, objPosCoordinates[1] - objSizeDenormalized[1] / 2],
			  [objPosCoordinates[0] + (float(objSizeDenormalized[0])), objPosCoordinates[1] + (float(objSizeDenormalized[1]))]]
	print("Object info is: ",objBox)
	
	yCoord1 = int(objBox[0][1])
	yCoord2 = int(objBox[1][1])
	xCoord1 = int(objBox[0][0])
	xCoord2 = int(objBox[1][0])

	if yCoord1 > 50:
		yCoord1 = yCoord1 - (random.random() * 40) + 10
	else:
		yCoord1 = 0
	if xCoord1 > 50:
		xCoord1 = xCoord1 - (random.random() * 40) + 10
	else:
		xCoord1 = 0
	output = image[int(yCoord1):int(yCoord2), int(xCoord1):int(xCoord2)]
	print("Final Image is: ", output.shape)
	#output_final = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
	return cv2.resize(output, OUTPUT_DIMENSIONS)

def saveSorted(img, info, counter, dir = "Dataset/Source_Materials/Sorted/"):
	for n in info:
		path = str(IMAGE_CLASSES[int(n[0])] + "/" +str(counter) + '.jpg')
		cv2.imwrite(path, cropInput(img, n))
		print("File saved at: ", path)
		print( "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")

textArray, imgArray = markDir()

for i, n in enumerate(textArray):
	info = readInfoFromFile(fileName = n)
	print("file name is: ", n)
	image = cv2.imread(str("Dataset/Source_Materials/Unsorted/" + imgArray[i]))
	saveSorted(image, info, i)

