import numpy as np
import pandas as pd
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample Normal Ranges
NORMAL_P_WAVE_DURATION = (0.08, 0.11)  # in seconds
NORMAL_QRS_DURATION = (0.06, 0.10)  # in seconds
NORMAL_T_WAVE_AMPLITUDE = (0.1, 0.5)  # in mV
NORMAL_HRV = (20, 100)  # Normal HRV in milliseconds

def extract_ecg_features(ecg_signal, sampling_rate):
    """Extract ECG features: P-wave, QRS duration, T-wave amplitude, and HRV."""
    r_peaks, _ = signal.find_peaks(ecg_signal, distance=sampling_rate * 0.6)
    rr_intervals = np.diff(r_peaks) / sampling_rate * 1000  # Convert to milliseconds
    
    # Mock Feature Extraction (Replace with actual algorithms)
    p_wave_duration = np.random.uniform(0.07, 0.12)  # Simulated Value
    qrs_duration = np.random.uniform(0.05, 0.14)  # Simulated Value
    t_wave_amplitude = np.random.uniform(0.05, 0.6)  # Simulated Value
    hrv = np.std(rr_intervals) if len(rr_intervals) > 1 else 0  # HRV Calculation
    
    return [p_wave_duration, qrs_duration, t_wave_amplitude, hrv]

def generate_dataset(n_samples=1000, sampling_rate=250):
    """Generate a synthetic ECG dataset with labels."""
    X, y = [], []
    for _ in range(n_samples):
        simulated_ecg = np.sin(np.linspace(0, 10 * np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
        features = extract_ecg_features(simulated_ecg, sampling_rate)
        label = np.random.choice([0, 1], p=[0.7, 0.3])  # 0: Healthy, 1: Diseased
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

def train_ecg_classifier():
    """Train a Random Forest classifier for ECG classification."""
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model

def predict_heart_disease(ecg_signal, model, sampling_rate=250):
    """Predict heart disease using a trained model."""
    features = extract_ecg_features(ecg_signal, sampling_rate)
    prediction = model.predict([features])[0]
    return "Healthy" if prediction == 0 else "High Risk"

# Train and Evaluate Model
trained_model = train_ecg_classifier()

# Test on a new ECG sample
new_ecg_signal = np.sin(np.linspace(0, 10 * np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
result = predict_heart_disease(new_ecg_signal, trained_model)
print("Diagnosis:", result)