import librosa
import numpy as np
import joblib

# Constants – these should match the settings used during training
MAX_LEN = 130
CUSTOM_THRESHOLD = 0.3  # Same custom threshold as in training
MODEL_PATH = "/Users/pragyamtiwari/bitcamp_website/classification/voice_classifier_ai_or_human.pkl"
SCALER_PATH = "/Users/pragyamtiwari/bitcamp_website/classification/scaler.pkl"

def extract_features(file_path, max_len=MAX_LEN):
    """
    Load an audio file and extract MFCC features.
    Pads or truncates the features to ensure a consistent shape.
    """
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        # Pad with zeros if the audio is too short, or truncate if too long
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def human_or_not(audio_file):
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        print(f"Failed to load model or scaler: {e}")
        return

    # Path to the audio file to classify
    features = extract_features(audio_file)

    if features is None:
        print("Feature extraction failed.")
        return

    # Reshape the feature array to match the model's expected input shape
    features = features.reshape(1, -1)

    # Apply the same normalization used during training
    features_normalized = scaler.transform(features)

    # Obtain prediction probabilities from the model
    # The probability at index 1 corresponds to the 'fake' (AI) class
    proba = model.predict_proba(features_normalized)
    ai_probability = proba[0][1]

    # Classify based on the custom threshold
    label = "AI" if ai_probability >= CUSTOM_THRESHOLD else "Human"
    
    print(f"Probability for AI: {ai_probability:.2f}")
    print(f"Classification result: {label}")

    return True if label == "Human" else False



