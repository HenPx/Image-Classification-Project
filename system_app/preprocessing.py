import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# Fungsi untuk melakukan cropping, mengubah ke grayscale, dan ekstraksi fitur GLCM
def preprocess_image(image_path):
    # Baca gambar
    image = cv2.imread(image_path)

    # Resize dan ubah gambar menjadi grayscale
    img = cv2.resize(image, (256, 256))
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    simpan = [grayscale_img]  # Menyimpan gambar yang telah diproses

    # Ekstrak fitur GLCM dari setiap sudut
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm_features = {}
    for angle in angles:
        glcm = graycomatrix(grayscale_img, [1], [angle], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        asm = graycoprops(glcm, 'ASM')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        glcm_features[angle] = [contrast, dissimilarity, homogeneity, asm, energy, correlation]
        print(f"GLCM untuk sudut {angle} adalah: {glcm_features[angle]}")

    return glcm_features, simpan
