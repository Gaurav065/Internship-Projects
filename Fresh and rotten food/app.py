import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

model_to_predict = tf.keras.models.load_model('chestmodel.hdf5')
def predict_covid(test_image):
    img = cv2.imread(test_image)
    img = img / 255.0
    img = cv2.resize(img, (224, 224))
    img = img.reshape(1,224,224,3)
    prediction = model_to_predict.predict(img)
    pred_class = np.argmax(prediction, axis = -1)
    return pred_class

def load_image(image_file):
    img = Image.open(image_file)
    return img


st.write("Food freshness using CNN")



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
        st.write('Fresh')
    elif pred[0] == 1:
        st.write('Fresh')
    elif pred[0] == 2:
        st.write('Fresh')
    elif pred[0] == 3:
        st.write('Rotten')
    elif pred[0] == 4:
        st.write('Rotten')
    elif pred[0] == 5:
        st.write('Rotten')