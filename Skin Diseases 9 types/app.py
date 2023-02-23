import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

model_to_predict = tf.keras.models.load_model('inceptionresnetv2_model (1).h5')
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


st.write("Skin diseases using InceptionResnetV2")



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
        st.write('Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions')
    elif pred[0] == 1:
        st.write('Atopic Dermatitis')
    elif pred[0] == 2:
        st.write('Eczema')
    elif pred[0] == 3:
        st.write('Melanoma Skin Cancer Nevi and Moles')
    elif pred[0] == 4:
        st.write('Nail Fungus and other Nail Disease')
    elif pred[0] == 5:
        st.write('Psoriasis pictures Lichen Planus and related diseases')
    elif pred[0] == 6:
        st.write('Seborrheic Keratoses and other Benign Tumors')
    elif pred[0] == 7:
        st.write('Tinea Ringworm Candidiasis and other Fungal Infections')
    elif pred[0] == 8:
        st.write('Warts Molluscum and other Viral Infections')