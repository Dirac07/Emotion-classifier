import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import keras
import sklearn
import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Input, BatchNormalization, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from keras.preprocessing import image
import scipy
import os
from PIL import Image
from tqdm import tqdm
import cv2
from numpy.lib.type_check import imag


def load_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("png files", "*.png")))
    basewidth = 150 # Processing image for dysplaying
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()


def detect_and_resize_faces(img):
    img = img.astype("uint8")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.01, 5)
    if len(faces) == 0:
        haar_img = img
    else:
        x, y, w, h = faces[0]
        haar_img = img[y:y+h, x:x+w]
        
    return cv2.resize(((haar_img).astype("float32")), (48,48))


def classify():
    original = Image.open(image_data).convert('L')
    numpy_image = detect_and_resize_faces(np.array(original))

    plt.imshow(numpy_image, cmap='gray')
    plt.show()

    predictions = model.predict(numpy_image.reshape(-1, 48, 48, 1)/255)[0]

    table = tk.Label(frame, text="Emotion predictions and confidences").pack()
    for i in range(0, len(emotions)):
         result = tk.Label(frame,
                    text= str(emotions[i]).upper() + ': ' + 
                           str(round(float(predictions[i])*100, 3)) + '%').pack()


root = tk.Tk()
root.title('Emotions Classifier')
root.resizable(False, False)
tit = tk.Label(root, text="Emotions Classifier", padx=25, pady=6, font=("", 12)).pack()
canvas = tk.Canvas(root, height=500, width=500, bg='grey')
canvas.pack()
frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
chose_image = tk.Button(root, text='Choose Image',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=load_img)
chose_image.pack(side=tk.LEFT)
class_image = tk.Button(root, text='Classify Image',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=classify)
class_image.pack(side=tk.RIGHT)


json_file = open('./model/model.json','r')
model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json(model_json)
model.load_weights("./model/model.h5")
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


root.mainloop()