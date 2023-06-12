import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

model_to_predict = tf.keras.models.load_model('updated.h5')

def predict_covid(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 3)
    prediction = model_to_predict.predict(img)
    pred_class = np.argmax(prediction, axis = -1)
    return pred_class

def load_image(image_file):
    img = Image.open(image_file)
    return img

def detect_plastic(img):
    pred = predict_covid(img)
    if pred[0] == 0:
        return 'cardboard'
    elif pred[0] == 1:
        return 'glass'
    elif pred[0] == 2:
        return 'metal'
    elif pred[0] == 3:
        return 'paper'
    elif pred[0] == 4:
        return 'plastic'
    elif pred[0] == 5:
        return 'trash'


