from shapedetector import ShapeDetector
import argparse
import imutils
import cv2

import numpy as np


def findBoundingBox(box, image_dimensions):
	topLeftCorner = 0
	bottomRightCorner = 0
	lowestXvalue = image_dimensions[1]
	lowestYvalue = image_dimensions[0]
	highestXvalue = 0
	highestYvalue = 0
	for point in box:
		if point[0] < lowestXvalue:
			lowestXvalue = point[0]
		elif point[0] > highestXvalue:
			highestXvalue = point[0]
		if point[1] < lowestYvalue:
			lowestYvalue = point[1]
		elif point[1] > highestYvalue:
			highestYvalue = point[1]

	lowestYvalue = lowestYvalue - 30
	if lowestYvalue < 0:
		lowestYvalue = 0
	lowestXvalue = lowestXvalue - 30
	if lowestXvalue < 0:
		lowestXvalue = 0
	highestYvalue = highestYvalue + 30
	if highestYvalue > image_dimensions[0]:
		highestYvalue = image_dimensions[0]
	highestXvalue = highestXvalue + 30
	if highestXvalue > image_dimensions[1]:
		highestXvalue = image_dimensions[1]

	topLeftCorner = [lowestXvalue, lowestYvalue]
	bottomRightCorner = [highestXvalue, highestYvalue]

	return (topLeftCorner, bottomRightCorner)

def thresholdImage(image, thresholdValue, thresholdWeight):
	adjusted = cv2.convertScaleAbs(image, alpha = 1.1, beta = 0)
	gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5 , 5), 0)
	thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	            cv2.THRESH_BINARY_INV,thresholdValue,thresholdWeight)
	return thresh

def saveForDataset(img, path = "Dataset/Source_Materials/Images/"):
	image = img.copy()
	dim = (32, 32)

	output = cv2.resize(image, dim)

	#cv2.imshow("Output Image", output)
	return output



# ap = argparse.ArgumentParser()

# ap.add_argument("-i", "--image", required = False, help = "Path to the input image")
# args = vars(ap.parse_args())

# img = cv2.imread(args["image"])

def detectShapes(img):

	image = img.copy()



	IMAGE_DIMENSIONS = image.shape[:2]
	resized = imutils.resize(image, width = 300)
	ratio = image.shape[0] / float(resized.shape[0])



	thresh = thresholdImage(resized.copy(), 1301, 7)
	#cv2.imshow("Threshold", thresh)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	cnts = imutils.grab_contours(cnts)
	result = []
	sd = ShapeDetector()
	for c in cnts:

		M = cv2.moments(c)
		if(M["m00"] == 0):
			shape = "line"
		else: 
			if cv2.contourArea(c) > 100 and cv2.contourArea(c) < 5000:		
				cX = int((M["m10"] / M["m00"]) * ratio)
				cY = int((M["m01"] / M["m00"]) * ratio)
				shape = sd.detect(c)
				c = c.astype("float")
				c *= ratio
				c = c.astype("int")
				rect = cv2.minAreaRect(c)
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
				cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				bBox = findBoundingBox(box,IMAGE_DIMENSIONS)
				imageForDataset = image[bBox[0][1]:bBox[1][1], bBox[0][0]:bBox[1][0]].copy()
				result.append(saveForDataset(imageForDataset))

				#cv2.imshow("Cropped Image", imageForDataset)
				#cv2.imshow("Image", image)
				#cv2.waitKey(2000)
	#cv2.waitKey(0)\

	result = np.stack(result, axis = 0)
	print(result.shape)
	return result, image

					
