
import streamlit as st
import pickle
import pandas as pd

# โหลดโมเดลและ encoders จากไฟล์ที่บันทึกไว้
with open('model_penguin_66130701908.pkl', 'rb') as f:
    model_data = pickle.load(f)

rf_model = model_data['model']
label_encoder_species = model_data['label_encoder_species']
label_encoder_sex = model_data['label_encoder_sex']

st.title("Penguin Species Prediction App")

# รับค่าคุณสมบัติจากผู้ใช้
culmen_length = st.number_input("Culmen Length (mm)", min_value=0.0, step=0.1)
culmen_depth = st.number_input("Culmen Depth (mm)", min_value=0.0, step=0.1)
flipper_length = st.number_input("Flipper Length (mm)", min_value=0.0, step=1.0)
body_mass = st.number_input("Body Mass (g)", min_value=0.0, step=1.0)
sex = st.selectbox("Sex", options=label_encoder_sex.classes_)

# เมื่อคลิกปุ่ม "Predict"
if st.button("Predict"):
    # แปลงข้อมูล sex เป็นตัวเลข
    sex_encoded = label_encoder_sex.transform([sex])[0]
    # รวมค่าคุณสมบัติทั้งหมดเป็น DataFrame
    input_data = pd.DataFrame([[culmen_length, culmen_depth, flipper_length, body_mass, sex_encoded]], 
                              columns=['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex'])
    # ทำนายผลลัพธ์
    prediction = rf_model.predict(input_data)
    species = label_encoder_species.inverse_transform(prediction)[0]
    
    st.write(f"The predicted species is: **{species}**")


