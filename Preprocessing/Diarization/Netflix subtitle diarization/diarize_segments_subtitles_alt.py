# -*- coding: utf-8 -*-
"""
Created on Sat May 27 18:44:32 2023

@author: nicol
"""

from pyAudioAnalysis import audioSegmentation as aS
from lxml import etree
import pandas as pd
from pydub import AudioSegment
import os
import json
import time

with open('secrets.json', 'r') as f:
    secrets = json.load(f)
HUGGING_FACE_APIKEY = secrets["HUGGING_FACE_APIKEY"]

audio_path = "D:/OneDrive - Universiteit Utrecht/Documents/000 - Documenti/PROJECTS/PSYCHE/TheOfficeSeries/Episodes/1/S01E01.mp3"
subtitles_path = "D:/OneDrive - Universiteit Utrecht/Documents/000 - Documenti/PROJECTS/PSYCHE/TheOfficeSeries/Subtitles/S01/S01E1.txt"
#********************* Note that the format has to be of the subtitles scraped from Netflix.

# Load and parse the XML file
tree = etree.parse(subtitles_path)
root = tree.getroot()

# Extract the subtitles
subtitles = []
for p in root.iter("{*}p"):
    start = int(p.get("begin")[:-1])/1e7  # Convert to seconds
    end = int(p.get("end")[:-1])/1e7  # Convert to seconds
    text = " ".join(p.itertext())
    subtitles.append({"start": start, "end": end, "text": text})

df_subs = pd.DataFrame(subtitles)

# Load audio file
audio = AudioSegment.from_mp3(audio_path)

# Initialize speaker column
df_subs['speaker'] = 'Unknown'

# Group subtitles every 50 sentences, ensuring those with the same end time stay together
groups = []
sentences_in_groups = 1000
for i in range(0, len(df_subs), sentences_in_groups):
    group = df_subs.iloc[i:i+sentences_in_groups]
    while i+sentences_in_groups < len(df_subs) and df_subs.iloc[i+sentences_in_groups-1]['end'] == df_subs.iloc[i+sentences_in_groups+1]['end']:
        group = df_subs.iloc[i:i+sentences_in_groups+1]
        i += 1
    groups.append(group)

for i, group in enumerate(groups):
    start = time.time()
    
    start_time = group['start'].min()
    end_time = group['end'].max()

    # Extract segment from audio file
    segment = audio[start_time * 1000:end_time * 1000]  # Pydub works with milliseconds

    print(f"Duration of the group: {len(segment) / 60000:.2f} minutes.")
    
    # Save segment to temporary file
    segment.export("Temp/temp.wav", format="wav")

    # Apply speaker diarization
    flags = aS.speaker_diarization("Temp/temp.wav", n_speakers=18)  # replace n_speakers with the number of speakers

    # Map diarized speakers to the subtitles in the group
    for index, speaker in enumerate(flags[0]): # index represents each 0.1 seconds
        df_subs.loc[(df_subs['start']*10 <= index) & (df_subs['end']*10 > index), 'speaker'] = speaker

    # Print progress
    end = time.time()
    # Print progress
    print(f"Finished diarization of group {i+1} of {len(groups)}. Time taken: {end-start:.2f} seconds.")
    
    break #!!!! delete this, it is only for testing

# Remove temporary file
#os.remove("Temp/temp.wav")

print(df_subs)
