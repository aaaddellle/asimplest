#%%
import librosa, librosa.display

import matplotlib.pyplot as plt
import numpy as np


#file = "C:\\Users\\Lenovo\\Desktop\\python\\AudioMNIST\\data\\01\\"
output_folder = "C:\\Users\\Lenovo\\Desktop\\python\\asimplest\\outputs"


def pad_audio(audio):
    current_length = len(audio)
    if current_length < 8000:
        padding_length = 8000 - current_length
        padding = np.zeros(padding_length)
        return np.concatenate((audio, padding))
    else:
        return audio[:8000]
    
def specto(filename):
    signal, sr = librosa.load(filename, sr = 22050)
    signal_padded = pad_audio(signal)

    librosa.display.waveshow(signal_padded, sr = sr)
    plt.xlabel("time")
    plt.ylabel("amplitude")

    #new

    '''folder_name = os.path.basename(filename).split("_")[1]
    output_subfolder = os.path.join(output_folder, folder_name)
    os.makedirs(output_subfolder, exist_ok=True)
    #new 
    #plt.show()
    #plt.savefig(os.path.join(output_folder, os.path.basename(filename)[:-4] + '.png'))
    output_filename = os.path.basename(filename)[:-4] + '.png'
    output_path = os.path.join(output_subfolder, output_filename)
    plt.savefig(output_path)

    plt.close()'''


     # Extract folder name from the filename
    folder_name = os.path.basename(os.path.dirname(filename))

    # Create the output subfolder if it doesn't exist
    output_subfolder = os.path.join(output_folder, folder_name)
    os.makedirs(output_subfolder, exist_ok=True)

    # Save the spectrogram image in the output subfolder
    output_filename = os.path.basename(filename)[:-4] + '.png'
    output_path = os.path.join(output_subfolder, output_filename)
    plt.savefig(output_path)

    plt.close()



import os


# Path to the main folder
main_folder = "C:\\Users\\Lenovo\\Desktop\\python\\asimplest\\audio"

# Loop through each folder in the main folder
for folder_name in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder_name)
    
    # Check if it's a directory
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder_name}")
        
        # Loop through each audio file in the current folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Check if it's a .wav file
            if file_name.endswith(".wav"):
                print(f"Processing audio file: {file_name}")
                
                # Generate spectrogram for the audio file
                specto(file_path)

print("Spectrograms saved successfully.")

'''
# Loop through each folder from '01' to '60'
for i in range(1, 10):
    # Convert the index to a folder name format (e.g., '01' instead of '1')
    folder_name = f"{i:02d}"
    folder_path = os.path.join(main_folder, folder_name)
    
    # Check if the folder exists
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder_name}")
        
        # Loop through each audio file in the current folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Check if it's a .wav file
            if file_name.endswith(".wav"):
                print(f"Processing audio file: {file_name}")
                
                # Generate spectrogram for the audio file
                specto(file_path)
'''
print("Spectrograms saved successfully.")
# %%
