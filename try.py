import os
import tkinter as tk
from tkinter import *

import numpy as np
import PIL
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from keras.models import load_model
from PIL import Image, ImageDraw, ImageOps, ImageTk
from tensorflow import keras

model = tf.saved_model.load('.')
print(model.trainable_variables)
