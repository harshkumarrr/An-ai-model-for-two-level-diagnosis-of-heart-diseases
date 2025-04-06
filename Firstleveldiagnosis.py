import numpy as np
import scipy.signal as signal

# Sample Normal Ranges
NORMAL_P_WAVE_DURATION = (0.08, 0.11)  # in seconds
NORMAL_QRS_DURATION = (0.06, 0.10)  # in seconds
NORMAL_T_WAVE_AMPLITUDE = (0.1, 0.5)  # in mV
NORMAL_HRV = (20, 100)  # Normal HRV in milliseconds

def compute_confidence(value, normal_range):
    """Calculate confidence factor based on normal range."""
    if normal_range[0] <= value <= normal_range[1]:
        return 1.0  # High Confidence
    elif abs(value - np.mean(normal_range)) < (normal_range[1] - normal_range[0]) * 0.5:
        return 0.5  # Medium Confidence
    else:
        return 0.0  # Low Confidence

def extract_ecg_features(ecg_signal, sampling_rate):
    """Simulated function to extract ECG features like P-wave, QRS complex, T-wave, and HRV."""
    
    # Detect R-peaks (For HRV and QRS duration estimation)
    r_peaks, _ = signal.find_peaks(ecg_signal, distance=sampling_rate * 0.6)  # Approx 60 BPM minimum
    rr_intervals = np.diff(r_peaks) / sampling_rate * 1000  # Convert to milliseconds
    
    # Mock Feature Extraction (Replace with actual algorithms)
    p_wave_duration = np.random.uniform(0.07, 0.12)  # Simulated Value
    qrs_duration = np.random.uniform(0.05, 0.14)  # Simulated Value
    t_wave_amplitude = np.random.uniform(0.05, 0.6)  # Simulated Value
    hrv = np.std(rr_intervals) if len(rr_intervals) > 1 else 0  # HRV Calculation
    
    return p_wave_duration, qrs_duration, t_wave_amplitude, hrv

def calculate_heart_disease_risk(ecg_signal, sampling_rate=250):
    """Computes confidence factors and final risk score based on ECG features."""
    
    # Extract ECG features
    p_wave_duration, qrs_duration, t_wave_amplitude, hrv = extract_ecg_features(ecg_signal, sampling_rate)
    
    # Compute Confidence Factors
    p_wave_confidence = compute_confidence(p_wave_duration, NORMAL_P_WAVE_DURATION)
    qrs_confidence = compute_confidence(qrs_duration, NORMAL_QRS_DURATION)
    t_wave_confidence = compute_confidence(t_wave_amplitude, NORMAL_T_WAVE_AMPLITUDE)
    hrv_confidence = compute_confidence(hrv, NORMAL_HRV)
    
    # Weighted Risk Score Calculation
    risk_score = (0.25 * p_wave_confidence +
                  0.35 * qrs_confidence +
                  0.20 * t_wave_confidence +
                  0.20 * hrv_confidence)
    
    return {
        "P-Wave Confidence": p_wave_confidence,
        "QRS Complex Confidence": qrs_confidence,
        "T-Wave Confidence": t_wave_confidence,
        "HRV Confidence": hrv_confidence,
        "Heart Disease Risk Score": risk_score,
        "Diagnosis": "Healthy" if risk_score > 0.8 else ("Borderline Risk" if risk_score > 0.5 else "High Risk")
    }

# Simulated ECG Signal (Replace with real data)
np.random.seed(42)
simulated_ecg = np.sin(np.linspace(0, 10 * np.pi, 1000)) + np.random.normal(0, 0.1, 1000)

# Compute Risk
result = calculate_heart_disease_risk(simulated_ecg)
print(result)
