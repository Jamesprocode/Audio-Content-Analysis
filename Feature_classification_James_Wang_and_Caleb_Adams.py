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
    num_files = len([f for f in os.listdir(folder) if f.endswith('.wav')])
    feature_matrix = np.zeros((num_files, num_features))
    
    # Storage for all frames across all files (for pre-aggregation normalization)
    all_flux_frames = []
    all_mfcc_frames = []
    
    file_list = [f for f in os.listdir(folder) if f.endswith('.wav')]
    
    # PASS 1: Extract features and collect all frames for flux and MFCC
    temp_features = []
    
    for index, file in enumerate(file_list):
        file_path = os.path.join(folder, file)
        
        # Read file
        audio_data, sr = librosa.load(file_path)
        # DC offset removal
        audio_data = audio_data - np.mean(audio_data)
        # Normalize amplitude
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        
        # Spectral spread
        spread = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        
        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=audio_data)
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio_data)
        
        # Spectral flux (with L2 normalization per frame)
        spectrogram = np.abs(librosa.stft(audio_data))
        L2_norms = np.linalg.norm(spectrogram, axis=0, keepdims=True)
        spectrogram_normalized = spectrogram / (L2_norms + 1e-10)  # Avoid division by zero
        flux_frames = np.sum(np.diff(spectrogram_normalized, axis=1) ** 2, axis=0)
        
        # MFCC (without normalization yet)
        mfcc_frames = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=10, n_mels=128)
        
        # Store for this file
        temp_features.append({
            'centroid': centroid,
            'spread': spread,
            'rolloff': rolloff,
            'flatness': flatness,
            'zcr': zcr,
            'flux_frames': flux_frames,
            'mfcc_frames': mfcc_frames
        })
        
        # Collect ALL frames across ALL files
        all_flux_frames.append(flux_frames)
        all_mfcc_frames.append(mfcc_frames)
    
    # PASS 2: Pre-aggregation normalization across ALL frames from ALL files
    all_flux_frames = np.concatenate(all_flux_frames)
    all_mfcc_frames = np.concatenate(all_mfcc_frames, axis=1)
    
    # Normalize flux: subtract mean, divide by std across all frames
    flux_mean_global = np.mean(all_flux_frames)
    flux_std_global = np.std(all_flux_frames)
    
    # Normalize MFCCs: CMVN across all frames (axis=1 is time)
    mfcc_mean_global = np.mean(all_mfcc_frames, axis=1, keepdims=True)
    mfcc_std_global = np.std(all_mfcc_frames, axis=1, keepdims=True)
    
    # PASS 3: Apply normalization and aggregate
    for index, features in enumerate(temp_features):
        # Aggregate features (mean across time)
        feature_matrix[index, 0] = np.mean(features['centroid'])
        feature_matrix[index, 1] = np.mean(features['spread'])
        feature_matrix[index, 2] = np.mean(features['rolloff'])
        feature_matrix[index, 3] = np.mean(features['flatness'])
        feature_matrix[index, 4] = np.mean(features['zcr'])
        
        # Normalize flux THEN aggregate
        flux_normalized = (features['flux_frames'] - flux_mean_global) / flux_std_global
        feature_matrix[index, 5] = np.mean(flux_normalized)
        
        # Normalize MFCCs THEN aggregate
        mfcc_normalized = (features['mfcc_frames'] - mfcc_mean_global) / mfcc_std_global
        mfcc_means = np.mean(mfcc_normalized, axis=1)
        feature_matrix[index, 6:16] = mfcc_means
    
    
    # Apply z-score normalization ACROSS FILES (for each feature column)
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    
  
    
    # Add all features to df
    feature_names = ['centroid_mean', 'spread_mean', 'rolloff_mean', 'flatness_mean', 
                     'zcr_mean', 'flux_mean'] + [f'mfcc_{i+1}_mean' for i in range(10)]
    
    for index, file in enumerate(file_list):
        for feat_idx, feat_name in enumerate(feature_names):
            df.loc[df['filename'] == file, feat_name] = feature_matrix[index, feat_idx]
    
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
    
  
    
    # Identify and report the highest and lowest correlations between any two features (excluding self-correlations). Create scatter plots for these two pairs of features, including a regression line, and appropriate titles and axis labels.
    # Get the feature columns (exclude filename and label columns)
    feature_cols = ['centroid_mean', 'spread_mean', 'rolloff_mean', 'flatness_mean', 
                    'zcr_mean', 'flux_mean'] + [f'mfcc_{i+1}_mean' for i in range(10)]


    # Find highest and lowest correlations (excluding diagonal)


    # Flatten and get indices
    correlations_flat = []
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):  # Only upper triangle
            correlations_flat.append({
                'feature1': feature_cols[i],
                'feature2': feature_cols[j],
                'correlation': correlation_matrix.iloc[i, j]
            })

    # Convert to dataframe for easier sorting
    corr_df = pd.DataFrame(correlations_flat)

    # Find highest correlation (closest to +1 or -1)
    corr_df['abs_correlation'] = np.abs(corr_df['correlation'])
    highest_corr = corr_df.loc[corr_df['abs_correlation'].idxmax()]
    lowest_corr = corr_df.loc[corr_df['abs_correlation'].idxmin()]

    print("="*60)
    print("HIGHEST CORRELATION:")
    print(f"{highest_corr['feature1']} vs {highest_corr['feature2']}")
    print(f"Correlation: {highest_corr['correlation']:.4f}")
    print("="*60)

    print("\nLOWEST CORRELATION:")
    print(f"{lowest_corr['feature1']} vs {lowest_corr['feature2']}")
    print(f"Correlation: {lowest_corr['correlation']:.4f}")
    print("="*60)

    # Create scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Highest correlation
    ax1 = axes[0]
    x_high = df[highest_corr['feature1']]
    y_high = df[highest_corr['feature2']]
    ax1.scatter(x_high, y_high, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
    ax1.set_xlabel(highest_corr['feature1'], fontsize=12)
    ax1.set_ylabel(highest_corr['feature2'], fontsize=12)
    ax1.set_title(f"Highest Correlation: r = {highest_corr['correlation']:.4f}", 
                fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add regression line
    z_high = np.polyfit(x_high, y_high, 1)
    p_high = np.poly1d(z_high)
    ax1.plot(x_high, p_high(x_high), "r--", alpha=0.8, linewidth=2, label='Linear fit')
    ax1.legend()

    # Plot 2: Lowest correlation
    ax2 = axes[1]
    x_low = df[lowest_corr['feature1']]
    y_low = df[lowest_corr['feature2']]
    ax2.scatter(x_low, y_low, alpha=0.6, s=50, edgecolors='k', linewidths=0.5, color='orange')
    ax2.set_xlabel(lowest_corr['feature1'], fontsize=12)
    ax2.set_ylabel(lowest_corr['feature2'], fontsize=12)
    ax2.set_title(f"Lowest Correlation: r = {lowest_corr['correlation']:.4f}", 
                fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add regression line
    z_low = np.polyfit(x_low, y_low, 1)
    p_low = np.poly1d(z_low)
    ax2.plot(x_low, p_low(x_low), "r--", alpha=0.8, linewidth=2, label='Linear fit')
    ax2.legend()

    plt.tight_layout()
    plt.show()
      

main()

