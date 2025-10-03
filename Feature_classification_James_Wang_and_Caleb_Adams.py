import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

def label_extraction(folder, df):
    #parse the label out of the file name
    pattern = re.compile(r'DB_validation-(\d+)_|DB_test-(\d+)_')  # strictly matches the number after DB_validation- or DB_test-

    for filename in os.listdir(folder):
        match = pattern.search(filename)
        if match:
            number = int(match.group(1) or match.group(2))  # extract the correct number
            df.loc[df['filename'] == filename, 'label'] = number

    df['label'] = df['label'].astype('Int64')  # Pandas nullable integer type
    return df

def feature_extraction(folder, df):
    num_features = 16
    feature_matrix = np.zeros((num_features, len(os.listdir(folder))))

    for index, file in enumerate(os.listdir(folder)):
        file_path = os.path.join(folder, file)
        if not file.endswith('.wav'):
            continue
        # Read file
        audio_data, sr = librosa.load(file_path)

        # DC offset
        audio_data = audio_data - np.mean(audio_data)
        # Normalize
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        centroid_mean = np.mean(centroid)
        feature_matrix[0][index] = centroid_mean

        # Spectral spread
        spread = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
        spread_mean = np.mean(spread)
        feature_matrix[1][index] = spread_mean

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        rolloff_mean = np.mean(rolloff)
        feature_matrix[2][index] = rolloff_mean

        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=audio_data)
        flatness_mean = np.mean(flatness)
        feature_matrix[3][index] = flatness_mean

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio_data)
        zcr_mean = np.mean(zcr)
        feature_matrix[4][index] = zcr_mean

        # Spectral flux
        spectrogram = np.abs(librosa.stft(audio_data))
        L2_norms = np.linalg.norm(spectrogram, axis=0)
        spectrogram = spectrogram / L2_norms #normalization
        flux = np.sum(np.diff(spectrogram, axis=1) ** 2, axis=0)
        flux_mean = np.mean(flux)
        feature_matrix[5][index] = flux_mean

        # MFCC
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mels=128)[:10]
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc) #normalization
        mfcc_means = np.mean(mfcc, axis=1)
        for mfcc_index, mfcc_mean in enumerate(mfcc_means):
          feature_matrix[6 + mfcc_index][index] = mfcc_mean

    # Apply z-normalization
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    # Add all features to df
    for index, file in enumerate(os.listdir(folder)):
        df.loc[df['filename'] == file, 'centroid_mean'] = feature_matrix[0][index]
        df.loc[df['filename'] == file, 'spread_mean'] = feature_matrix[1][index]
        df.loc[df['filename'] == file, 'rolloff_mean'] = feature_matrix[2][index]
        df.loc[df['filename'] == file, 'flatness_mean'] = feature_matrix[3][index]
        df.loc[df['filename'] == file, 'zcr_mean'] = feature_matrix[4][index]
        df.loc[df['filename'] == file, 'flux_mean'] = feature_matrix[5][index]
        for mfcc_index, mfcc_mean in enumerate(mfcc_means):
          df.loc[df['filename'] == file, f'mfcc_{mfcc_index+1}_mean'] = feature_matrix[6 + mfcc_index][index]

    return df

def main():
    
#getting the file
    folder = 'audios/micro_medlydb/test'
    test_folder = 'audios/micro_medlydb/test'

    #initialize the matrix with names of the audio files in the directory
    audio_files = np.array([f for f in os.listdir(folder) if f.endswith('.wav')])
    test_files = np.array([f for f in os.listdir(test_folder) if f.endswith('.wav')])

    #create a numpy matrix to store the names of the audio files
    df = pd.DataFrame(audio_files, columns=['filename'])
    df_test = pd.DataFrame(test_files, columns=['filename'])

    #extract the labels from the file names and store them in the dataframe
    df = label_extraction(folder, df)
    df_test = label_extraction(test_folder, df_test)

    #feature extraction
    df = feature_extraction(folder, df)
    df_test = feature_extraction(test_folder, df_test)

    #Compute a correlation matrix accross the calculated features (i.e., each column of features is compared to every other column of features). Report the matrix.
    #creating a new dataframe with only the features
    feature_df = df.drop(columns=['filename', 'label'])
    correlation_matrix = feature_df.corr()
    
  
    print(df.head())
    print(df_test.head())


main()

