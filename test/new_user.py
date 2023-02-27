from image_similarity_api import get_distance
import streamlit as st
import os
from PIL import Image
import pandas as pd
import uuid
import warnings
from send_mail import SendMail
import requests

os.makedirs(os.path.join('data','user_data','found_images'),exist_ok=True)
th = 0.8

def gps_tracker():
    r = requests.get('https://ipinfo.io/?token=6c098641e84c13')
    return r.json()

import os
import streamlit as st
import pandas as pd

def load_image(image_file):
    img = Image.open(image_file)
    return img

def view_missing_image(image_file):
    with open(image_file, 'rb') as f:
        img_bytes = f.read()
    st.image(img_bytes, use_column_width=True)

def display_missing_reports():
    csv_path = os.path.join('data','admin_data','missing_data.csv')
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['img_path'])
    st.dataframe(df)
    
    if st.checkbox('View missing person image'):
        row = st.selectbox('Select a missing person:', df.index)
        image_file = df.loc[row, 'img_path']
        view_missing_image(image_file)
    
# main function
def main():
    page = st.sidebar.radio(label="Menu",options=['Report Missing Person','Missing Reports from admin'])

    if page == "Report Missing Person":
        st.markdown("> Report a Missing Person")
        person_name = st.text_input(label="Name")
        person_image = st.file_uploader("Upload Missing Person Image Here", type=["jpg","jpeg"])
        contact_number = st.number_input(label="Phone Number",value=976)
        submit_button = st.button('Submit')

        if submit_button:
            if person_name and person_image:
                # save the image
                img_path = os.path.join('data','admin_data','missing_images',f'{person_name}.jpg')
                img_data = load_image(person_image)
                img_data.save(img_path)

                # save the details in CSV file
                csv_path = os.path.join('data','admin_data','missing_data.csv')
                data = {'name': [person_name],
                        'contact_number': [contact_number],
                        'img_path': [img_path],
                        'Match-found': ['No'],
                        'email_sent': ['No']}
                df = pd.DataFrame(data)
                if os.path.exists(csv_path):
                    df.to_csv(csv_path, mode='a', header=False, index=False)
                else:
                    df.to_csv(csv_path, index=False)

                st.write('Missing person report submitted!')
            else:
                st.write('Please provide a name and an image for the missing person.')

    if page == 'Missing Reports from admin':
        st.markdown("> Missing Data Reports")
        display_missing_reports()

if __name__ == '__main__':
    main()
