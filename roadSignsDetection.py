import tensorflow as tf
import os
import sys
import numpy
import cv2
import tkinter as tk

from dataset import rSigns
from model import ModelWrapper
from detect_shapes import detectShapes 

window = tk.Tk()
window.title("Model Demo")

trainButton = tk.Button(window, text = "train", activebackground = "lightgray",  width = 15, command = window.destroy)
trainButton.grid()

window.mainloop()