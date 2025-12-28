import os
import glob
import pickle
import librosa
import soundfile
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Feature extraction function
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

# Adult emotions (RAVDESS)
emotions = {
    '01':'neutral', '02':'calm', '03':'happy',
    '04':'sad', '05':'angry', '06':'fearful',
    '07':'disgust', '08':'surprised'
}
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Baby cry emotions
BABY_EMOTIONS = {
    'hungry': 0,
    'tired': 1,
    'pain': 2,
    'discomfort': 3,
    'burping': 4
}

def load_adult_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob("Dataset/all_audio/*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion in observed_emotions:
            x.append(extract_feature(file))
            y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

def load_baby_data(test_size=0.2):
    x, y = [], []
    for emotion, code in BABY_EMOTIONS.items():
        for file in glob.glob(f"Dataset/babycrydataset/{emotion}/*.wav"):
            x.append(extract_feature(file))
            y.append(code)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

def train_voice_classifier():
    # Label 0 for adult, 1 for baby
    X, y = [], []
    
    # Add adult samples
    for file in glob.glob("Dataset/all_audio/*.wav"):
        X.append(extract_feature(file))
        y.append(0)
    
    # Add baby samples
    for emotion in BABY_EMOTIONS.keys():
        for file in glob.glob(f"Dataset/babycrydataset/{emotion}/*.wav"):
            X.append(extract_feature(file))
            y.append(1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
    model = MLPClassifier(alpha=0.01, batch_size=256, hidden_layer_sizes=(200,), max_iter=500)
    model.fit(X_train, y_train)
    print(f"Voice Classifier Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
    return model

if __name__ == "__main__":
    # Train voice classifier first
    voice_classifier = train_voice_classifier()
    pickle.dump(voice_classifier, open("Pickle_Voice_Classifier.pkl", "wb"))
    
    # Train adult model
    x_train, x_test, y_train, y_test = load_adult_data(test_size=0.25)
    adult_model = MLPClassifier(alpha=0.01, batch_size=256, hidden_layer_sizes=(300,), max_iter=500)
    adult_model.fit(x_train, y_train)
    
    # Train baby model
    bx_train, bx_test, by_train, by_test = load_baby_data(test_size=0.25)
    baby_model = MLPClassifier(alpha=0.01, batch_size=256, hidden_layer_sizes=(300,), max_iter=500)
    baby_model.fit(bx_train, by_train)
    
    # Save models
    pickle.dump(adult_model, open("Pickle_RL_Model.pkl", "wb"))
    pickle.dump(baby_model, open("Pickle_Baby_Model.pkl", "wb"))
    
    # Print accuracies
    print(f"Adult Model Accuracy: {accuracy_score(y_test, adult_model.predict(x_test)):.2f}")
    print(f"Baby Model Accuracy: {accuracy_score(by_test, baby_model.predict(bx_test)):.2f}")