import os
import pandas as pd
import numpy as np
import librosa

def featureExtractor(path='./AutSED_New_Name_selectedData', SR=44100):
    files = os.listdir(path)
    np.random.shuffle(files)
    audio_length = int(5*SR)

    label_symbol = {'H':0, 'A':1, 'S':2, 'N':3}
    feature_stack = []
    label_stack = []
    for f in files:
        label_stack.append(label_symbol[f[3]])
        audio, SR = librosa.load(os.path.join(path, f), sr=44100)

        if len(audio) < audio_length:
            audio = np.concatenate((audio, np.zeros(audio_length - len(audio))))
        elif len(audio) > audio_length:
            audio = audio[:audio_length]

        feature = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=20)
        feature_stack.append(feature)
        #print(feature.shape)

    feature_stack = np.stack(feature_stack, axis=0)
    #print(feature_stack.shape)
    return feature_stack, np.array(label_stack)

def onehotEncoder(label):
    assert len(label.shape)==1, print('input size error, size:', label.shape)
    one_hot_label = np.zeros((len(label), np.max(label)+1), dtype=np.int8)
    for i, j in zip(label, one_hot_label):
        j[i] = 1
    return one_hot_label

