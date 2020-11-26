import numpy as np
import cv2
import argparse, os
import random

def validate_file(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

counter = "0"

def saveAll():
	global counter
	variableName = args.output_directory + str(counter) + ".jpg"
	counter = int(counter) + 1
	cv2.imwrite(variableName, image)

def saveRandom():
	global counter
	randomValue = random.random()
	randomValue = int(randomValue * 15)
	if randomValue == 1:
		variableName = args.output_directory + str(counter) + ".jpg"
		counter = int(counter) + 1
		print(counter)
		cv2.imwrite(variableName, image)

parser = argparse.ArgumentParser(description = "Extract frames as separate images from video")
parser.add_argument('--type', '-t', dest="extractType", help = "Type of frames selection", default = 'all')
parser.add_argument('--input', '-i',dest="input_filename", type=validate_file, metavar = "FILE", required = True, help = "Name of the video to extract from")
parser.add_argument('--output', '-o',dest="output_directory", metavar = "FILE", help = "Folder to extract to")
args = parser.parse_args()
print(args.input_filename)
videoStream = cv2.VideoCapture(args.input_filename)

onlyfiles = [f for f in os.listdir(args.output_directory) if os.path.isfile(os.path.join(args.output_directory, f))]
print(len(onlyfiles))
even = True
contents = os.listdir(args.output_directory)
counter = str(int(counter) + len(onlyfiles))
currentFrame = 0
allFrames = videoStream.get(int(cv2.CAP_PROP_FRAME_COUNT))
print(cv2.CAP_PROP_FRAME_COUNT)
while(True):
	ret, frame = videoStream.read()
	currentFrame = currentFrame + 1
	if ret == False:
		break
	progress = currentFrame / allFrames * 100

	kernel = np.array([[-1, -1, -1], [-1, 9,-1], [-1, -1, -1]])
	image = cv2.filter2D(frame, -1, kernel)
	if args.extractType == 'all':
		saveAll()
	elif args.extractType == 'even':
		if even == True:
			saveAll()
			even = False
		else:
			even = True
	elif args.extractType == 'rand':
		saveRandom()
		print("{:4d}%".format(progress))

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
videoStream.release()
cv2.destroyAllWindows()

