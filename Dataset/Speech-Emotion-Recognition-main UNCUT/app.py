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
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Empty file'}), 400

        if not audio_file.filename.endswith('.wav'):
            return jsonify({'error': 'Only .wav files allowed'}), 400

        temp_path = "temp.wav"
        audio_file.save(temp_path)
        features = extract_feature(temp_path)
        
        # First classify voice type
        voice_type = voice_classifier.predict([features])[0]
        
        # Then get appropriate prediction
        if voice_type == 0:  # Adult
            emotion = adult_model.predict([features])[0]
            result = {'type': 'adult', 'emotion': emotion}
        else:  # Baby
            emotion_code = baby_model.predict([features])[0]
            emotion = BABY_EMOTIONS.get(emotion_code, 'unknown')
            result = {'type': 'baby', 'emotion': emotion}
        
        os.remove(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)