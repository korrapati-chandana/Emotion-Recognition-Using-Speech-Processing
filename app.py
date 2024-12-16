from flask import Flask, render_template, request, jsonify
import joblib
import librosa
import numpy as np
import os
from tensorflow.keras.models import load_model

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and label encoder
model = load_model('emotion_model.h5')
encoder = joblib.load('label_encoder.joblib')

# Function to extract MFCC features
def extract_features(file_path, sr=22050, n_mfcc=40):
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Route to handle the home page and file upload
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")

    # Save the uploaded file temporarily
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Extract features and predict the emotion
    features = extract_features(file_path)
    if features is not None:
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        emotion_index = np.argmax(prediction)
        emotion = encoder.classes_[emotion_index]
        
        # Clean up the uploaded file
        os.remove(file_path)
        
        return render_template('index.html', prediction=emotion)
    else:
        return render_template('index.html', error="Error extracting features from the audio")

if __name__ == "__main__":
    # Create an uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(debug=True)


  