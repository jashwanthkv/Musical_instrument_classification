import os
import librosa
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify,url_for
import joblib  # For loading saved models
from sklearn.ensemble import VotingClassifier

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained models and scaler
rf = joblib.load('.venv/random_forest.pkl')  # Replace with the actual path
svm = joblib.load('.venv/SVM.pkl')  # Replace with the actual path
scaler = joblib.load('.venv/scaler (1).pkl')  # Replace with the actual path

# Define a Voting Classifier that uses the RF and SVM
voting_clf = joblib.load('.venv/votingclf.pkl')

# Instrument Image Dictionary
instrument_images = {
    "Flute": "Flute.jpg",
    "BassDraw": "BassDraw (1).jpg",
    "Cello": "Cello.jpg",
    "Clarinet": "Clarinet.jpg.jpg",
    "DoubleBass": "DoubleBass.jpg.jpg",
    "Hihat": "Hihat.jpg.jpg",
    "Kamancheh": "Kamancheh.jpg",
    "Ney": "Ney.jpg",
    "Santur": "Santur.jpg.jpg",
    "Setar": "Setar.jpg.jpg",
    "Saxophone": "Saxophone.jpg",
    "Tar": "Tar.jpg",
    "Violin": "Violin.jpg"
    # Add other instruments and their image paths here
}


# Function to extract features from an audio file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        # Extract features
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)

        # Combine features into a single array
        features = np.hstack(
            [mfccs, chroma, mel_spectrogram, spectral_contrast, tonnetz, spectral_centroid, spectral_bandwidth])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# Define route for file upload
@app.route('/')
def index():
    return render_template('index.html')


# Handle the file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    # Check if the file is a valid .wav file
    if file and file.filename.endswith('.wav'):
        # Save the uploaded file
        file_path = os.path.join('.venv/uploads', file.filename)
        file.save(file_path)

        # Extract features from the uploaded file
        features = extract_features(file_path)

        if features is not None:
            # Prepare the features for prediction
            feature_columns = (
                    [f"mfcc_{i + 1}" for i in range(13)] +
                    [f"chroma_{i + 1}" for i in range(12)] +
                    [f"mel_{i + 1}" for i in range(128)] +
                    [f"spectral_contrast_{i + 1}" for i in range(7)] +
                    [f"tonnetz_{i + 1}" for i in range(6)] +
                    ["spectral_centroid", "spectral_bandwidth"]
            )

            # Create DataFrame for the extracted features
            df = pd.DataFrame([features], columns=feature_columns)

            # Extract features for prediction
            X_unseen = df.values

            # Step 1: Get class probabilities from the pre-trained Random Forest model
            rf_probs = rf.predict_proba(X_unseen)

            # Step 2: Combine original features with RF probabilities
            X_unseen_combined = np.hstack((X_unseen, rf_probs))

            # Step 3: Standardize the combined features using the pre-trained scaler
            X_unseen_scaled = scaler.transform(X_unseen_combined)

            # Step 4: Predict with the pre-trained Voting Classifier
            y_pred = voting_clf.predict(X_unseen_scaled)

            # Get the predicted label (instrument name)
            predicted_label = y_pred[0]

            # Get the image path corresponding to the predicted label
            image_path = url_for('static', filename=instrument_images.get(predicted_label, "default.png"))
            return jsonify(
                {"predicted_label": predicted_label, "image_path": image_path, "probabilities": rf_probs.tolist()})

        else:
            return jsonify({"error": "Error extracting features from the file"}), 400
    else:
        return jsonify({"error": "Invalid file format. Please upload a .wav file."}), 400


# Run the Flask app
if __name__ == '__main__':
    # Ensure the 'uploads' directory exists
    if not os.path.exists('.venv/uploads'):
        os.makedirs('.venv/uploads')

    app.run(debug=True)
