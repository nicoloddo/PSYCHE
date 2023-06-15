# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:22:59 2023

@author: nicol

Transforms into filelist from a segmentation_df with the columns: 
    [wav_path, transcription_path, emotion]
"""

import pandas as pd
import os
import re

import __libpath
from psychelibrary import psyche_tools as pt

def replace_words(transcription, disfluency_dict):
    for word, new_word in disfluency_dict.items():
        pattern = r'\b{}\b'.format(word)  # \b is a word boundary in regex
        transcription = re.sub(pattern, new_word, transcription)
    return transcription

def preprocess(transcription):
    transcription = replace_words(transcription, pt.disfluency_dict)
    return transcription

def load_transcription(transcription_path):
    # Check if file exists
    if os.path.isfile(transcription_path):
        with open(transcription_path, 'r') as f:
            return f.read().strip()
    else:
        return None

def filelist(df, data_folder, output_file):
    # Add the prefix 'data/IEMOCAP/' to 'wav_path'.
    df['wav_path'] = df['wav_path'].apply(lambda x: data_folder + os.path.basename(x))

    print("Loading transcriptions...")
    # Apply the loading function to the transcription column.
    df['transcription'] = df['transcription_path'].apply(load_transcription)
    print(df.info())
    
    print("Dropping nan...")
    # Remove rows with None in transcription after reading from files
    df = df.dropna(subset=['transcription'])
    print(df.info())
    
    print("Preprocessing transcriptions...")
    # Preprocess the transcriptions
    df['transcription'] = df['transcription'].apply(preprocess)
    
    print("Labeling emotions...")
    # Replace emotions with corresponding integer values.
    df['emotion'] = df['emotion'].replace(pt.emotion_dict)

    # Take only the desired columns.
    df = df[['wav_path', 'transcription', 'emotion']]

    # Convert DataFrame to string and save it to a text file.
    df.to_csv(output_file, header=None, index=None, sep='|', line_terminator='\n')


data_folder = 'data/IEMOCAP'
output_file = '../Save/IEMOCAP/iemocap_filelist.txt'
df= pd.read_pickle("../Save/IEMOCAP/segmented.pkl")

filelist(df, data_folder, output_file)




