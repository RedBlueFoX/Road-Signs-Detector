# Road Signs Detector
 Artificial Intelligence Course Project

##Final Report

###Contents

1. Technological Stack
2. Design
3. Structure
4. Pipeline
5. Code Explanation
6. Dataset
7. UI
8. User Guide
9. Conclusions

###Technological Stack

**Frameworks:**
```
python --version
Python 3.8.6

tensorflow.__version__
'2.5.0-dev20201128'

flask --version
Python 3.8.6
Flask 1.1.2
Werkzeug 1.0.1

cv2.__version__
'4.4.0'
```

**Scripts:**

*Zoorjz's Fake Images Generator for objects recognition:* https://github.com/Zoorjz/FakeImagesGenerator_obj_recognition

###Design

The original idea lies in using OpenCV capabilites to work with images in order to extract shapes from the image, thus simplifying the work required to detect an object on the image. In theory and it works fine, however in order to work properly it requires a gradient descent implemented what should improve both accuracy and speed, because OpenCV is much faster and lighter than Tensorflow Model that is fed with random samples from the original image. However, by increasing the computational power running Tensorflow on bigger batch could prove of being faster.

Design of the Tensorflow was based on multiple researchers and combines some basic techniques as well as some advanced tecniques.

*Structure of Tensorflow model:*
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 32)        128       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 32)        9248      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 32)        1056      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 14, 14, 32)        9248      
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 12, 12, 64)        18496     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 10, 10, 128)       73856     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 128)         0         
_________________________________________________________________
flatten (Flatten)            (None, 3200)              0         
_________________________________________________________________
dropout (Dropout)            (None, 3200)              0         
_________________________________________________________________
dense (Dense)                (None, 1024)              3277824   
_________________________________________________________________
dense_1 (Dense)              (None, 256)               262400    
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 96)                24672     
=================================================================
```

**I'll start from the beginning**:
1. I supply 32 by 32 image with 3 channels into the Sequential Keras model.
2. First 3 layers create a technique called the "bottleneck block" originally created by ResNet. The main idea behind this is impoving computational costs and accuracy at the same time by reducing the use amount of 3x3 feature maps that tensorflow needs to create from 1024 to 32. However i'm pretty sure that for my model the numbers are way off and I should have used way bigger numbers for this. Later i will explain why.
3.After running the "bottleneck block" i perform maxPooling and do simple gradient feature extraction.
4.After convolutions are done I have simple NN with two hidden dense layers to train on the data.
5. The output is 96 nodes for each label

*Technically, it can run the video object detection, because it is not really hardware-heavy.*
###Structure:
####Important Part
#####Dataset
Folder with dataset source images 
#####Dataset.py
Script that is required to prepare dataset for the loading
#####Detect_shapes.py
Script that extract shapes from the image
#####Model
Folder with model checkpoints
#####Model.py
Wrapper for the model for ease of use
#####Shapedetector.py
Utility script for Detect_shapes.py proper operation
#####UIWeb.py
A Flask script to host a web-server
#####Uploads, Static, Templates, Venv
Folders for the Flask web-server

###Pipeline

*I'll describe that process step-by-step, including training:*

1. Loading images from the disk
2. Processing them for the later use
3. Train the dataset with really small learning rate to prevent over-fitting.
4. Save the dataset using callbacks after each epoch.

*Traning is done:*

1. Load model or use pretrained one.
2. Feed the image into OpenCV script that returns the image with boundary boxes around all the objects it has found and the array of the images that are resized for model evaluation.
3. Feed them into the model.
4. Receive the prediction.
5. Parse and sort them from highest probability to lowest within certain threshold
6. Output them.

###Code explanation

*If i will have time before 12 p.m i will come back if not i will explain during presentation.*

###Dataset

**I used VideoBreaker.py script to extract random frames from videos of cars driving aroung the city to use as a backgrounds. Then I created by hand a set of 4 augmented images for each label by hand in photoshop. I used Zoorjz script to generate 200 images for each label by adding the images onto the backgrounds with random augmentations.**

###UI

I run Flask web-server to host a web-page with the simplest interface possible. I have a choose file button in system language which can be confusing but it is not customizable and I have an upload button. When you upload the file it is sent to the script that feeds it into the pipeline and awaits for the results. The results are written into the uploads folder because that's the easiest way to send files back.

###User Guide

1. Navigate to the project folder in the command prompt
2. Set the environment variable FLASK_APP=UIWeb.py
3. Run the command "flask run"
4. Open your web-browser and visit page "http://127.0.0.1:5000"
5. Upload a file 
6. Analyse the results


###Conclusions

1. The model converges too fast. Even tho i lowered the learning rate and after 95-98 iterations it showed perfected results, after testing on real photos the results were disappointing. I think the issue is with the convolution filters, theirs amount and mostly the size of the images in dataset. I should've ran with at least 64x64.
2. The dataset loading time is very huge. There are few ways how i think that could be improved like hashing the data into the one file.
3. UI was the hardest part of all of the coding.
4. I could've combined both OpenCV shapes detection and random sampling. Probably if i did multiple shapes detection sequentially and after that some random sampling that would improve the detection algorithm but it comes back to the precision of the model. If the model can't provide accurate result of what is and what isn't a road sign, i can't evaluate an algorithm appropriately.