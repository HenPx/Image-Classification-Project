import numpy as np
import joblib

# Mengimpor model yang telah disimpan
best_modelSVM_0 = joblib.load("system_app/model/model_sudut_0.pkl")
best_modelSVM_45 = joblib.load("system_app/model/model_sudut_45.pkl")
best_modelSVM_90 = joblib.load("system_app/model/model_sudut_90.pkl")
best_modelSVM_135 = joblib.load("system_app/model/model_sudut_135.pkl")

# Fungsi untuk melakukan prediksi
def predict_ripeness(glcm_features):
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    score = 0
    predictions = {}
    for angle in angles:
        fitur = np.array([glcm_features[angle]])
        if angle == 0:
            prediction = best_modelSVM_0.predict(fitur)
            predictions['sudut_0'] = prediction[0]
            if prediction == 'matang':
                score += 1
            elif prediction == 'busuk':
                score -= 1
        elif angle == np.pi/4:
            prediction = best_modelSVM_45.predict(fitur)
            predictions['sudut_45'] = prediction[0]
            if prediction == 'matang':
                score += 1
            elif prediction == 'busuk':
                score -= 1
        elif angle == np.pi/2:
            prediction = best_modelSVM_90.predict(fitur)
            predictions['sudut_90'] = prediction[0]
            if prediction == 'matang':
                score += 1
            elif prediction == 'busuk':
                score -= 1
        elif angle == 3*np.pi/4:
            prediction = best_modelSVM_135.predict(fitur)
            predictions['sudut_135'] = prediction[0]
            if prediction == 'matang':
                score += 1
            elif prediction == 'busuk':
                score -= 1

    if score > 0:
        final_prediction = "matang"
    elif score < 0:
        final_prediction = "busuk"
    else:
        final_prediction = "ambigu"

    return final_prediction, predictions
