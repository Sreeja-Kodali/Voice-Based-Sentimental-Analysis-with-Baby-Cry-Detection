from flask import Flask, request, jsonify, send_file
import pickle
import librosa
import soundfile
import numpy as np
import os

app = Flask(__name__)

# Load all models
with open("Pickle_Voice_Classifier.pkl", "rb") as f:
    voice_classifier = pickle.load(f)
with open("Pickle_RL_Model.pkl", "rb") as f:
    adult_model = pickle.load(f)
with open("Pickle_Baby_Model.pkl", "rb") as f:
    baby_model = pickle.load(f)

# Baby emotions reverse mapping
BABY_EMOTIONS = {
    0: 'hungry',
    1: 'tired',
    2: 'pain',
    3: 'discomfort',
    4: 'burping'
}

def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))  # 40 Features

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))  # 12 Features

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))  # 128 Features

    return result  # Total: 40 + 12 + 128 = 180 Features

    

@app.route('/predict', methods=['POST'])
def predict():
    temp_path = "temp.wav"
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Empty file uploaded'}), 400

        if not audio_file.filename.endswith('.wav'):
            return jsonify({'error': 'Only .wav files are allowed'}), 400

        audio_file.save(temp_path)
        features = extract_feature(temp_path)

        if features is None or len(features) == 0:
            return jsonify({'error': 'Feature extraction failed'}), 500

        # First classify voice type
        voice_type = voice_classifier.predict([features])[0]
        print(f"Voice Type Prediction: {voice_type}")  # Debugging print

        # Then get appropriate prediction
        if voice_type == 0:  # Adult
            emotion_prob = adult_model.predict_proba([features])[0]  # Get confidence scores
            emotion = adult_model.predict([features])[0]
            print(f"Adult Emotion: {emotion}, Probabilities: {emotion_prob}")  # Debugging print
            result = {'type': 'adult', 'emotion': emotion, 'confidence_scores': emotion_prob.tolist()}
        else:  # Baby
            emotion_code_prob = baby_model.predict_proba([features])[0]  # Get confidence scores
            emotion_code = baby_model.predict([features])[0]
            emotion = BABY_EMOTIONS.get(emotion_code, 'unknown')
            print(f"Baby Emotion: {emotion}, Probabilities: {emotion_code_prob}")  # Debugging print
            result = {'type': 'baby', 'emotion': emotion, 'confidence_scores': emotion_code_prob.tolist()}

        return jsonify(result)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)  # Ensure file cleanup

@app.route('/')
def index():
    return send_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)