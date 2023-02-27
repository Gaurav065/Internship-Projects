import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

model_to_predict = tf.keras.models.load_model('garbage_plastic_inceptionResnetv2.h5')
def predict_covid(test_image):
    img = cv2.imread(test_image)
    img = img / 255.0
    img = cv2.resize(img, (128, 128))
    img = img.reshape(1,128,128,3)
    prediction = model_to_predict.predict(img)
    pred_class = np.argmax(prediction, axis = -1)
    return pred_class

def load_image(image_file):
    img = Image.open(image_file)
    return img


st.write("Plastic detection using InceptionResnetV2")



pic = st.file_uploader("Upload a picture!")
submit = st.button('submit')



if submit:
    pic_details = {"filename":pic.name, 'filetype':pic.type, 'filesize':pic.size}
    st.write(pic_details)

    st.image(load_image(pic), width=250)

    with open('test.jpg', 'wb') as f:
        f.write(pic.getbuffer())
    pred = predict_covid('test.jpg')
    if pred[0] == 0:
        st.write('its not plastic but cardboard')
    elif pred[0] == 1:
        st.write('its not plastic but glass')
    elif pred[0] == 2:
        st.write('its not plastic but metal')
    elif pred[0] == 3:
        st.write('its not plastic but paper')
    elif pred[0] == 4:
        st.write('Its plastic')
    elif pred[0] == 5:
        st.write('its not plastic but trash and that to random')