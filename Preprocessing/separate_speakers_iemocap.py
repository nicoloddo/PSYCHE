# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 22:18:01 2023

@author: nicol
"""

import pandas as pd
from pydub import AudioSegment
import noisereduce as nr
import numpy as np
from scipy.io.wavfile import read, write
from pydub.playback import play

# Here's a function that processes one row
def process_audio(row):
    # Load the audio file
    audio = AudioSegment.from_wav(row['wav_path'])

    # Cut the audio to the required duration
    start_time_ms = row['start_time'] * 1000  # start_time should be in seconds
    end_time_ms = row['end_time'] * 1000  # end_time should be in seconds
    audio = audio[start_time_ms:end_time_ms]

    # Split stereo to mono
    channels = audio.split_to_mono()

    # Denoise each channel
    denoised_channels = []
    for i, channel in enumerate(channels):
        # Export channel to a temporary file
        channel.export("Temp/temp" + str(i+1) + ".wav", format="wav")
        
    # Read wav file to array
    rate1, data1 = read("Temp/temp1.wav")
    rate2, data2 = read("Temp/temp2.wav")
    
    # Perform noise reduction
    #reduced_noise1 = nr.reduce_noise(y=data1, y_noise=data2, sr=rate1) #!!! with the y_noise label it seems to perform worse
    reduced_noise1 = nr.reduce_noise(y=data1, sr=rate1)
    reduced_noise2 = nr.reduce_noise(y=data2, sr=rate2) 
    
    # Write back to wav
    write("Save/test_denoised1.wav", rate1, np.array(reduced_noise1, dtype=np.int16))
    write("Save/test_denoised2.wav", rate2, np.array(reduced_noise2, dtype=np.int16))
    
    # Read back denoised audio
    #denoised_audio = AudioSegment.from_wav("Temp/temp_denoised.wav")
    
    # Append denoised audio to list
    #denoised_channels.append(denoised_audio)
        
    #return denoised_channels

dialog = pd.read_pickle("Save/dialog.pkl")
subset = dialog.loc[dialog['conversation_id'] == 'Ses01F_impro03']
subset['processed_audio'] = subset.apply(process_audio, axis=1)
