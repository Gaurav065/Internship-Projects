import streamlit as st
import os
import pickle
from pathlib import Path
import streamlit_authenticator as stauth  
from PIL import Image
import pandas as pd
import warnings

os.system('python generate_keys.py')
df = pd.DataFrame(columns=['name','age','img_path','Contact_no','Location','email','Match-found','email_sent'])
os.makedirs(os.path.join('data','admin_data','missing_images'),exist_ok=True)
csv_file_path = os.path.join('data','admin_data','missing_data.csv')
if not os.path.isfile(csv_file_path):
    df.to_csv(csv_file_path,index=False)
del df

def load_image(image_file,path_to_save):
    img = Image.open(image_file)
    path_to_save = path_to_save+'.'+img.format.lower()
    img.save(path_to_save)
    return img,path_to_save

def save_admin_data(img,name,age,num,email,location):
    img_path = os.path.join('data','admin_data','missing_images',f'{name}')
    img_data,img_path = load_image(img,path_to_save=img_path)
    missing_person_data = {'name':name,'age':age,'img_path':img_path,'Contact_no':num,'Location':location,'email':email,'Match-found':'No','email_sent':'No'} 
    df2 = pd.DataFrame(missing_person_data,index=[0])
    csv_path = os.path.join('data','admin_data','missing_data.csv')
    df = pd.read_csv(csv_path)
    # df = df.append(missing_person_data, ignore_index = True).reset_index(drop=True) 
    df = pd.concat([df, df2], ignore_index = True)
    df.to_csv(csv_path,index=False)
    return img_data

# --- USER AUTHENTICATION ---

# ...

# --- USER AUTHENTICATION ---

def get_username(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        admin_logins = f.readlines()
    admin_logins = [i.strip() for i in admin_logins]
    names = list(filter(lambda x: 'names' in x,admin_logins))
    names = names[0].split('=')[1].strip()
    usernames = list(filter(lambda x: 'usernames' in x,admin_logins))
    usernames = usernames[0].split('=')[1].strip()
    return eval(names),eval(usernames)

names,usernames = get_username('generate_keys.py')

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "admin", "abcdef", cookie_expiry_days=0)

if not authenticator.is_authenticated():
    st.warning("Please log in to continue")
    if "register" not in st.session_state:
        st.session_state.register = False
    if not st.session_state.register:
        st.write("To access this app, you need to be registered.")
        register = st.button("Register")
        if register:
            st.session_state.register = True

if authenticator.is_authenticated():
    authenticator.logout("Logout", "sidebar")
    page = st.sidebar.radio(label="Menu",options=['Report Missing Person','Reports from Users'])
    if page == "Report Missing Person":
        st.markdown("> Report a Missing Person")
        person_name = st.text_input(label="Name")
        person_image = st.file_uploader("Upload Missing Person Image Here", type=["jpg","jpeg"])
        contact_number = st.number_input(label="Phone Number",value=9769123443)
        age = st.number_input(label="age",min_value=1)
        email = st.text_input(label="Email address")
        location = st.text_input(label="Enter Location")
        report = st.button(label="Report")
        if report:
            img = save_admin_data(person_image,person_name,age,contact_number,email,location)
            st.image(img)
            st.markdown("Missing Data Report Saved")
    if page == "Reports from Users":
        st.markdown("> Missing Data Reports")
        csv_path = os.path.join('data','admin_data','missing_data.csv')
        df = pd.read_csv(csv_path)
        st.dataframe(df)
        for i in range(df.shape[0]):
            image_path = df.iloc[i]['img_path']
            st.write(f"Image for {df.iloc[i]['name']}")
            image = Image.open(image_path)
            st.image(image)
