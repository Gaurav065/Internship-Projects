import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

model_to_predict = tf.keras.models.load_model('garbage_plastic_inceptionResnetv2.h5')

def predict_covid(img):
    img = cv2.resize(img, (128, 128))
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

st.write("Plastic detection using InceptionResnetV2")

option = st.selectbox('Select mode', ['Image', 'Video'])

if option == 'Image':
    pic = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    submit = st.button('Detect')

    if submit and pic is not None:
        st.image(load_image(pic), width=250)
        img = np.array(load_image(pic))
        pred = detect_plastic(img)
        st.write(f'The object in the image is {pred}.')

elif option == 'Video':
    vid = st.file_uploader("Upload a video", type=["mp4", "mov"])
    submit = st.button('Detect')

    if submit and vid is not None:
        video_bytes = vid.read()
        st.video(video_bytes)

        # OpenCV requires a byte object to read the video file
        video_file = np.asarray(bytearray(video_bytes), dtype=np.uint8)
        cap = cv2.VideoCapture(video_file)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect the object in the frame
            pred = detect_plastic(frame)
            
            # Display the frame and the detected object label
            cv2.putText(frame, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            st.image(frame, channels='BGR', use_column_width=True)

        cap.release()
