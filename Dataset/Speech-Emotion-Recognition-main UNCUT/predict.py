import pickle, glob, subprocess
from sys import argv

# function to install a library using pip3
def install(name):
    subprocess.call(['pip3', 'install', name])

# installing required libraries
try:
    import librosa
except:
    install('librosa')
    import librosa

try:
    import soundfile
except:
    install('soundfile')
    import soundfile

try:
    import numpy as np
except:
    install('numpy')
    import numpy as np

Pkl_Filename = "Pickle_RL_Model.pkl"  

with open(Pkl_Filename, 'rb') as file:  
    model = pickle.load(file)

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result


if len(argv)>1:
    inputs = []
    
    for input in argv[1:]:
        inputs.append(extract_feature(input))
    
    predicted_emotions = model.predict(inputs)

    for file, emotion in zip(argv[1:], predicted_emotions):
        print('File {} predicted to have a {} emotion.'.format(file, emotion))

else:
    print(model)
    