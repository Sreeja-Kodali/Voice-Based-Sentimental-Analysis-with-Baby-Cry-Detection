# importing standard libraries
import subprocess, os, glob, pickle, sys, random, pickle

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

try:
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
except:
    install('sklearn')
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
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
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)  # Corrected line
            result=np.hstack((result, mel))
    return result

 # Emotions in the RAVDESS dataset
emotions={
'01':'neutral',
'02':'calm',
'03':'happy',
'04':'sad',
'05':'angry',
'06':'fearful',
'07':'disgust',
'08':'surprised'
}

arguments = []
result = []
name = []

# Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']

# Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x, y = [], []

    audio_files = glob.glob("Dataset/*/*.wav")

    random.shuffle(audio_files)

    for file in audio_files:
        file_name = os.path.basename(file)

        emotion = emotions[file_name.split("-")[2]]

        # Remove the problematic line:
        # file_name = file[file.rindex('/') + 1:]

        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)

        if file_name in arguments and file_name not in name:
            result.append(feature)
            name.append(file_name)
            continue

        x.append(feature)
        y.append(emotion)

    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)
# main function
if __name__ == "__main__":
    arguments = sys.argv[1:]
    max_accuracy = 0.00000000000

    while True:
        x_train, x_test, y_train, y_test = load_data(test_size=0.25)
        model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            print('Shape of test and train datsets respectively', (x_train.shape[0], x_test.shape[0]))
            print(f'Features extracted: {x_train.shape[1]}')
            print("Accuracy: {:.2f}%".format(accuracy * 100))

            if result: # check if the result list is not empty
                result_array = np.array(result) # convert result list to a numpy array.
                resultant_emotions = model.predict(result_array)

                ind = 0
                for res, emo in zip(result, resultant_emotions):
                    print('{} file is predicted as {} emotion'.format(name[ind], emo))
                    ind += 1
            else:
                print("No files to predict from command line arguments.")

            Pkl_Filename = "Pickle_RL_Model.pkl"
            with open(Pkl_Filename, 'wb') as file:
                pickle.dump(model, file)
            print()

            if accuracy >= 80:
                break