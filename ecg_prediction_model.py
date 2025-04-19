import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import scipy.signal as signal

# Constants for normal ranges
NORMAL_P_WAVE_DURATION = (0.08, 0.11)
NORMAL_QRS_DURATION = (0.06, 0.10)
NORMAL_T_WAVE_AMPLITUDE = (0.1, 0.5)
NORMAL_HRV = (20, 100)

# Disease mapping
DISEASE_LABELS = {
    0: "Arrhythmia",
    1: "Myocardial Infarction",
    2: "Heart Block",
    3: "Atrial Fibrillation"
}

# --- IMAGE TO SIGNAL ---
def extract_ecg_signal_from_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1000, 300))  # Resize for consistency
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)
    signal_row = np.mean(binary, axis=0)
    signal_row = signal_row - np.min(signal_row)
    return signal_row / np.max(signal_row)

# --- FEATURE EXTRACTION ---
def extract_ecg_features(ecg_signal, sampling_rate=250):
    r_peaks, _ = signal.find_peaks(ecg_signal, distance=sampling_rate * 0.6)
    rr_intervals = np.diff(r_peaks) / sampling_rate * 1000

    p_wave_duration = np.random.uniform(0.07, 0.12)
    qrs_duration = np.random.uniform(0.05, 0.14)
    t_wave_amplitude = np.random.uniform(0.05, 0.6)
    hrv = np.std(rr_intervals) if len(rr_intervals) > 1 else 0

    return [p_wave_duration, qrs_duration, t_wave_amplitude, hrv]

# --- CONFIDENCE CALCULATION ---
def calculate_confidence(features):
    p_wave, qrs, t_wave, hrv = features
    confidence_factors = {}

    if not (NORMAL_P_WAVE_DURATION[0] <= p_wave <= NORMAL_P_WAVE_DURATION[1]):
        confidence_factors["Abnormal P-Wave"] = "Possible atrial enlargement or conduction delay."
    if not (NORMAL_QRS_DURATION[0] <= qrs <= NORMAL_QRS_DURATION[1]):
        confidence_factors["Abnormal QRS Complex"] = "May indicate bundle branch block or ventricular hypertrophy."
    if not (NORMAL_T_WAVE_AMPLITUDE[0] <= t_wave <= NORMAL_T_WAVE_AMPLITUDE[1]):
        confidence_factors["Abnormal T-Wave Amplitude"] = "Can be a sign of ischemia, electrolyte imbalance, or myocardial infarction."
    if not (NORMAL_HRV[0] <= hrv <= NORMAL_HRV[1]):
        confidence_factors["Abnormal HRV"] = "Low HRV suggests autonomic dysfunction; high HRV may indicate arrhythmia."

    confidence_score = 1 - (len(confidence_factors) / 4)
    return confidence_score, confidence_factors

# --- SYNTHETIC DATASET GENERATION ---
def generate_dataset(n_samples=1000):
    X, y = [], []
    for _ in range(n_samples):
        signal_data = np.sin(np.linspace(0, 10 * np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
        features = extract_ecg_features(signal_data)
        label = np.random.randint(0, len(DISEASE_LABELS))  # Simulated disease labels
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

# --- TRAINING MODELS ---
def train_models():
    X, y = generate_dataset()
    y_binary = np.where(y == 0, 0, 1)  # Healthy (0) vs Diseased (1)

    # Split once and reuse for both models
    X_train, X_test, y_binary_train, y_binary_test, y_multiclass_train, y_multiclass_test = train_test_split(
        X, y_binary, y, test_size=0.2, random_state=42
    )

    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_rf.fit(X_train, y_binary_train)

    clf_xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
    clf_xgb.fit(X_train, y_multiclass_train)

    # (Optional: print evaluation here)

    return clf_rf, clf_xgb


# --- PREDICT FROM IMAGE ---
def predict_from_image(image_path, model, xgb_model):
    ecg_signal = extract_ecg_signal_from_image(image_path)
    features = extract_ecg_features(ecg_signal)
    prediction = model.predict([features])[0]
    confidence_score, confidence_factors = calculate_confidence(features)

    if prediction == 0:
        diagnosis = "Healthy"
        specific_disease = "N/A"
    else:
        disease_code = xgb_model.predict([features])[0]
        specific_disease = DISEASE_LABELS.get(int(disease_code), "Unknown Disease")
        diagnosis = "High Risk"

    return {
        "Diagnosis": diagnosis,
        "Confidence Score": round(confidence_score, 2),
        "Factors": confidence_factors,
        "Specific Disease": specific_disease
    }

# --- RUN ---
if __name__ == "__main__":
    rf_model, xgb_model = train_models()

    test_image_path = "Normal(34).jp"  # <-- Replace with your ECG image path
    result = predict_from_image(test_image_path, rf_model, xgb_model)
    print("Diagnosis Report from Image:", result)



