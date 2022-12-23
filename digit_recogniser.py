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
from scipy import stats as st
from tensorflow import keras

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
width = 500
height = 500
center = height//2
white = (255, 255, 255)
green = (0, 128, 0)


class CNN(nn.Module):

    def __init__(self, num_filters1, num_filters2):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(
            in_channels=1, out_channels=num_filters1, kernel_size=3, stride=1, padding='same')
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn2 = nn.Conv2d(
            in_channels=num_filters1, out_channels=num_filters2, kernel_size=3, stride=1, padding='same')
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(49*num_filters2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


class cnn(tf.Module):

    def __init__(self, in_features, out_features):
        super(cnn, self).__init__()
        self.W_cnn1 = tf.Variable(tf.random.normal([3, 3, 1, in_features], stddev=0.1),
                                  name="w_cnn1")
        self.b_cnn1 = tf.Variable(tf.constant(
            0.1, shape=[in_features]), name="b_cnn2")
        self.W_cnn2 = tf.Variable(tf.random.normal([3, 3, in_features, out_features], stddev=0.1),
                                  name="w_cnn2")
        self.b_cnn2 = tf.Variable(tf.constant(
            0.1, shape=[out_features]), name="b_cnn2")
        self.w1 = tf.Variable(tf.random.normal([1568, 256], stddev=0.1),
                              name="w1")
        self.b1 = tf.Variable(tf.random.normal([1, 256], stddev=0.1),
                              name="b1")
        self.w2 = tf.Variable(tf.random.normal([256, 64], stddev=0.1),
                              name="w2")
        self.b2 = tf.Variable(tf.random.normal([1, 64], stddev=0.1), name="b2")
        self.w3 = tf.Variable(tf.random.normal([64, 10], stddev=0.1),
                              name="w3")
        self.b3 = tf.Variable(tf.random.normal([1, 10], stddev=0.1), name="b3")

    def __call__(self, x):
        x = tf.nn.conv2d(
            x, filters=self.W_cnn1, padding='SAME', strides=[1, 1, 1, 1
                                                             ]) + self.b_cnn1
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=(2, 2), strides=(2, 2), padding="VALID")
        x = tf.nn.conv2d(
            x, filters=self.W_cnn2, padding='SAME', strides=[1, 1, 1, 1
                                                             ]) + self.b_cnn2
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=(2, 2), strides=(2, 2), padding="VALID")
        x = tf.reshape(x, [-1, 1568])
        x = tf.matmul(x, self.w1) + self.b1
        x = tf.nn.relu(x)
        x = tf.matmul(x, self.w2) + self.b2
        x = tf.nn.relu(x)
        x = tf.matmul(x, self.w3) + self.b3

        return x


def mode(lst):

    # creating a dictionary
    freq = {}
    for i in lst:

        # mapping each value of list to a
        # dictionary
        freq.setdefault(i, 0)
        freq[i] += 1

    # finding maximum value of dictionary
    hf = max(freq.values())

    # creating an empty list
    hflst = []

    # using for loop we are checking for most
    # repeated value
    for i, j in freq.items():
        if j == hf:
            hflst.append(i)

    # returning the result
    return hflst


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def predict_digit(type):
    if type == 'keras':
        model = keras.models.load_model('digit_keras.h5')
    elif type == 'torch':
        model = torch.load('digit_torch.pt')
        model.eval()
    elif type == 'tensorflow':
        model = tf.saved_model.load('.')
    image = Image.open('image.png')
    image = image.resize((28, 28))
    image = image.convert('L')
    image = ImageOps.invert(image)
    image = np.asarray(image).astype(np.float32)/255
    if type == 'keras':
        image = image.reshape((1, 28, 28, 1))
        y_pred = model.predict(image, verbose=0)
    elif type == 'torch':
        image = image.reshape((1, 1, 28, 28))
        y_pred = model(torch.from_numpy(image))
        y_pred = y_pred.detach().numpy()
    elif type == 'tensorflow':
        image = image.reshape((1, 28, 28, 1))
        y_pred = model(image)
        y_pred = y_pred.numpy()
    y_pred = np.array(softmax(y_pred[0]))
    return np.array([y_pred])


def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    cv.create_oval(x1, y1, x2, y2, fill="black", width=30)
    draw.line([x1, y1, x2, y2], fill="black", width=30)


def model_keras():
    filename = "image.png"
    image1.save(filename)
    pred = predict_digit('keras')
    print('argmax', np.argmax(pred[0]), '\n',
          pred[0][np.argmax(pred[0])], '\n', classes[np.argmax(pred[0])])
    txt.insert(
        tk.INSERT, f"{classes[np.argmax(pred[0])]}\nAccuracy: {round(pred[0][np.argmax(pred[0])]*100, 3)}%\n")


def model_torch():
    filename = "image.png"
    image1.save(filename)
    pred = predict_digit('torch')
    print('argmax', np.argmax(pred[0]), '\n',
          pred[0][np.argmax(pred[0])], '\n', classes[np.argmax(pred[0])])
    txt.insert(
        tk.INSERT, f"{classes[np.argmax(pred[0])]}\nAccuracy: {round(pred[0][np.argmax(pred[0])]*100, 3)}%\n")


def model_tensorflow():
    filename = "image.png"
    image1.save(filename)
    pred = predict_digit('tensorflow')
    print('argmax', np.argmax(pred[0]), '\n',
          pred[0][np.argmax(pred[0])], '\n', classes[np.argmax(pred[0])])
    txt.insert(
        tk.INSERT, f"{classes[np.argmax(pred[0])]}\nAccuracy: {round(pred[0][np.argmax(pred[0])]*100, 3)}%\n")


def model_best():
    filename = "image.png"
    image1.save(filename)
    pred = predict_digit('keras')
    pred1 = predict_digit('torch')
    pred2 = predict_digit('tensorflow')
    arr = [classes[np.argmax(pred[0])],
           classes[np.argmax(pred1[0])], classes[np.argmax(pred2[0])]]
    prediction = mode(arr)
    prob = round(pred[0][np.argmax(pred[0])]*100, 3) + round(pred1[0]
                                                             [np.argmax(pred1[0])]*100, 3) + round(pred2[0][np.argmax(pred2[0])]*100, 3)
    prob = prob/3
    print('argmax', np.argmax(pred[0]), '\n',
          pred[0][np.argmax(pred[0])], '\n', classes[np.argmax(pred[0])])
    txt.insert(
        tk.INSERT, f"{prediction[0]}\nAccuracy: {prob}%\n")


def clear():
    cv.delete('all')
    draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))
    txt.delete('1.0', END)


root = Tk()
# root.geometry('1000x500')

root.resizable(0, 0)
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()

# PIL create an empty image and draw object to draw on
# memory only, not visible
image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

txt = tk.Text(root, bd=3, exportselection=0, bg='WHITE', font='Helvetica',
              padx=10, pady=10, height=5, width=20)

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

# button=Button(text="save",command=save)
btnModel_keras = Button(text="Predict_Keras", command=model_keras)
btnModel_torch = Button(text="Predict_Torch", command=model_torch)
btnModel_tensorflow = Button(
    text="Predict_Tensorflow", command=model_tensorflow)
btnModel_best = Button(text="Predict_Best", command=model_best)
btnClear = Button(text="clear", command=clear)
# button.pack()
btnModel_keras.pack()
btnModel_torch.pack()
btnModel_tensorflow.pack()
btnModel_best.pack()
btnClear.pack()
txt.pack()
root.mainloop()
