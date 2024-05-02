import librosa, librosa.display

import matplotlib.pyplot as plt
import numpy as np

import os
import numpy as np

def generate_mel_spectrogram(audio_file_path, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128, window_size_sec=0.025, overlap=0.5):
    # Load the audio file
    y, sr = librosa.load(audio_file_path, sr=sample_rate, dtype=np.float32)
    
    if len(y.shape) > 1:
        y = librosa.to_mono(y)
    if sr != sample_rate:
        y = librosa.resample(y, sr, sample_rate)
        sr = sample_rate

    window_length = int(window_size_sec * sr)
    hop_length = int(window_length * (1 - overlap))

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram_db
    '''mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Convert to decibels (log scale)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram_db'''

def plot_mel_spectrogram(mel_spectrogram, sample_rate=22050):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    #plt.show()

# Define directories
data_dir = 'C:\\Users\\Lenovo\\Desktop\\python\\asimplest\\audio'
output_dir = 'C:\\Users\\Lenovo\\Desktop\\python\\asimplest\\mels_1'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
'''
# Iterate through all folders and audio files
for folder_name in sorted(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder_name)
    if os.path.isdir(folder_path):
        output_folder_path = os.path.join(output_dir, folder_name)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                output_file_path = os.path.join(output_folder_path, file_name.replace('.wav', '.png'))
                mel_spectrogram = generate_mel_spectrogram(file_path)
                plot_mel_spectrogram(mel_spectrogram)
                plt.savefig(output_file_path)
                plt.close()'''

for folder_name in sorted(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder_name)
    if os.path.isdir(folder_path):
        output_folder_path = os.path.join(output_dir, folder_name)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                output_file_path = os.path.join(output_folder_path, file_name.replace('.wav', '.png'))
                mel_spectrogram = generate_mel_spectrogram(file_path)
                plot_mel_spectrogram(mel_spectrogram)
                plt.savefig(output_file_path)
                plt.close()  # Close the plot after saving