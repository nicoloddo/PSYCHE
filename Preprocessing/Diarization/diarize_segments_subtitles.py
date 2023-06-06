# -*- coding: utf-8 -*-
"""
Created on Sat May 27 13:42:59 2023

@author: nicol

Segments and diarize audio labeling the subtitles to a speaker code. Note that the subtitles has to be in the format of the ones scraped from Netflix.
"""

from pyannote.audio import Pipeline
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
    start = int(p.get("begin")[:-1])
    end = int(p.get("end")[:-1])
    text = " ".join(p.itertext())
    subtitles.append({"start": start, "end": end, "text": text})

df_subs = pd.DataFrame(subtitles)

# Load audio file
audio = AudioSegment.from_mp3(audio_path)

# Initialize pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HUGGING_FACE_APIKEY)

# Map the speakers to the subtitles
df_subs['start'] /= 1e7  # Convert start time to seconds
df_subs['end'] /= 1e7  # Convert end time to seconds

df_subs['speaker'] = 'Unknown'

# Group subtitles every 50 sentences
groups = df_subs.groupby(df_subs.index // 50)

for i, group in groups:
    start = time.time()
    start_time = group['start'].min()
    end_time = group['end'].max()

    # Extract segment from audio file
    segment = audio[start_time * 1000:end_time * 1000]
    
    print(f"Duration of the group: {len(segment) / 60000:.2f} minutes.")
    
    # Save segment to temporary file
    segment.export("Temp/temp.wav", format="wav")

    # Apply the pipeline to the audio segment
    diarization = pipeline("Temp/temp.wav", min_speakers=2, max_speakers=5)

    # Map diarized speakers to the subtitles in the group
    for segment, _, speaker in diarization.itertracks(yield_label=True):
            df_subs.loc[(df_subs['start'] >= segment.start) & (df_subs['end'] <= segment.end), 'speaker'] = speaker

    end = time.time()
    # Print progress
    print(f"Finished diarization of group {i+1} of {len(groups)}. Time taken: {end-start:.2f} seconds.")
    
    break #!!!! delete this, it is only for testing

# Remove temporary file
#os.remove("Temp/temp.wav")

print(df_subs)
