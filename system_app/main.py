import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import preprocess_image
from prediction import predict_ripeness
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Prediksi Kelayakan Buah",
    page_icon="🍏",
)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Judul
with st.container():
    st.markdown("<h1 class='stTitle'>Prediksi Kelayakan Buah Golden Apple</h1>", unsafe_allow_html=True)

with st.container():
    st.markdown("<h2 class='stSubHeader'>Kelompok C4</h2>", unsafe_allow_html=True)

st.image("system_app/asset/hero.jpg", use_column_width=True, width=100)

with st.container():
    st.markdown("<h6 class='stSubHeader2'>Apel Emas (Golden Apple) merupakan salah satu varietas apel yang sangat populer di seluruh dunia karena rasa manisnya yang khas, tekstur renyah, serta warnanya yang menarik. Apel ini pertama kali diperkenalkan di Amerika Serikat pada awal abad ke-20 dan sejak itu telah menjadi salah satu pilihan utama di antara konsumen dan petani. </h6>", unsafe_allow_html=True)

with st.container():
    st.markdown("<h1 class='stSubHeaderPred'>Prediksi Golden Apple Anda 🤤</h1>", unsafe_allow_html=True)


# Unggah gambar
uploaded_files = st.file_uploader("Unggah gambar buah", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key='file_uploader')

for uploaded_file in uploaded_files:
    # Simpan gambar yang diunggah
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    original_image = Image.open(uploaded_file)
    # Preproses gambar
    glcm_features, processed_image = preprocess_image("temp_image.jpg")

    st.image(original_image, caption='Gambar Asli yang Diunggah', use_column_width=True)
    
    # Tampilkan gambar yang diproses
    st.image(processed_image[0], caption='Gambar yang Diproses', use_column_width=True, channels="GRAY")
    
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    angle_labels = [int(np.degrees(angle)) for angle in angles]
    
    # Tampilkan DataFrame fitur GLCM per sudut
    dataframes = {}
    columns = ['Contrast', 'Dissimilarity', 'Homogeneity', 'ASM', 'Energy', 'Correlation']

    for angle in angles:
        df = pd.DataFrame([glcm_features[angle]], columns=columns)
        degree = np.degrees(angle)  # Konversi radian ke derajat
        dataframes[int(degree)] = df
        st.markdown(f"<div class='glcm-title'>DataFrame untuk sudut {int(degree)}°:</div>", unsafe_allow_html=True)
        st.write(df)

    # Membuat grafik dari fitur GLCM
    fig, ax = plt.subplots(figsize=(10, 6))
    for angle, label in zip(angles, angle_labels):
        df = dataframes[int(np.degrees(angle))]  # Mengambil DataFrame berdasarkan sudut dalam derajat
        ax.plot(columns, df.iloc[0], marker='o', label=f'Sudut {label}°')
    ax.set_title('Fitur GLCM per Sudut')
    ax.set_xlabel('Fitur')
    ax.set_ylabel('Nilai')
    ax.legend()
    st.pyplot(fig)
    
    # Lakukan prediksi
    final_prediction, predictions = predict_ripeness(glcm_features)
    
    # Tampilkan hasil prediksi
    st.write("Hasil prediksi per sudut:")
    for angle, pred in predictions.items():
        st.write(f"Hasil prediksi untuk sudut {angle}: {pred}")
    
    st.success(f"Hasil prediksi akhir: {final_prediction}")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: grey; font-size: 0.8em;'>© 2024 Prediksi Kelayakan Buah Golden Apple- KC4 Server. All rights reserved.</div>", unsafe_allow_html=True)