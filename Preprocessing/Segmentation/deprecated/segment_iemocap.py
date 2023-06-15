# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 18:43:24 2023

@author: nicol
"""

import json
from pydub import AudioSegment
import os
import pandas as pd

import __libpath
from psychelibrary import psyche_dataset as psd
from psychelibrary import psyche_tools as pt

# Path to JSON files
ALIGN_DIR = 'D:/OneDrive - Universiteit Utrecht/Documents/000 - Documenti/PROJECTS/PSYCHE/Dataset/IEMOCAP_full_release_withoutVideos.tar/IEMOCAP_full_release_withoutVideos/IEMOCAP_full_release/alignments/'

alignment_id = 'iemocap_default'
json_path = ALIGN_DIR + alignment_id + '/breath/json'

save_folders = [
    ALIGN_DIR + alignment_id + '/breath/segmented_alignment/',
    ALIGN_DIR + alignment_id + '/breath/segmented_alignment/json/',
    ALIGN_DIR + alignment_id + '/breath/segmented_tokens/',
    ALIGN_DIR + alignment_id + '/breath/segmented_wav/',
    ]

for folder in save_folders:
    pt.create_save_folder(folder)
    
# Path to the output transcriptions and JSON files
transcription_path = save_folders[0]
segmented_json_path = save_folders[1]
segmented_wav_path = save_folders[3]

# Threshold for splitting longer turns in milliseconds
threshold = 30000  # Adjust this value as needed

# Your dictionary of conversation_ids and wav paths
content = pd.read_pickle("../Save/content.pkl")
psyche = psd.PsycheDataset('IEMOCAP', content)
conv_dict = psyche.audios

# Loop through all JSON files in the directory
for filename in os.listdir(json_path):
    if filename.endswith(".json"):
        # Open the JSON file and load the content
        with open(os.path.join(json_path, filename)) as f:
            data = json.load(f)

        # Create a dictionary to hold words for each turn
        turns = {}
        for word in data['words']:
            if 'turn' in word:
                if word['turn'] not in turns:
                    turns[word['turn']] = []
                turns[word['turn']].append(word)
            elif word['word'] == '<breath>':
                # Add breath to the previous turn and the next turn if they are different
                for turn in turns:
                    if turns[turn] and turns[turn][-1]['end'] == word['start']:
                        turns[turn].append(word)
                        break

        conversation_id = os.path.splitext(filename)[0]
        audio = AudioSegment.from_wav(conv_dict[conversation_id])

        # Process each turn
        for turn, words in turns.items():
            # Segment the audio
            segments = []
            for word in words:
                segments.append(audio[word['start']*1000: word['end']*1000])

            # Combine all segments into one
            combined = sum(segments)

            # If the combined segment is longer than the threshold, split it
            if len(combined) > threshold:
                # Calculate the number of splits needed
                num_splits = len(combined) // threshold
                if len(combined) % threshold != 0:
                    num_splits += 1

                # Split the combined segment
                splits = [combined[i*threshold:(i+1)*threshold] for i in range(num_splits)]

                # Save each split as a separate file, and save its transcription and JSON
                for i, split in enumerate(splits):
                    split_name = f"{turn}_{i}"
                    split.export(os.path.join(segmented_wav_path, f"{split_name}.wav"), format="wav")

                    # Save transcription
                    transcription = ' '.join([word['word'] for word in words[i*threshold:(i+1)*threshold]])
                    with open(os.path.join(transcription_path, f"{split_name}.txt"), "w") as f:
                        f.write(transcription)

                    # Save segmented JSON
                    segmented_json = {"words": words[i*threshold:(i+1)*threshold]}
                    with open(os.path.join(segmented_json_path, f"{split_name}.json"), "w") as f:
                        json.dump(segmented_json, f, indent=4)
            else:
                # Save the combined segment, and save its transcription and JSON
                combined_name = f"{turn}"
                combined.export(os.path.join(segmented_wav_path, f"{combined_name}.wav"), format="wav")

                # Save transcription
                transcription = ' '.join([word['word'] for word in words])
                with open(os.path.join(transcription_path, f"{combined_name}.txt"), "w") as f:
                    f.write(transcription)

                # Save segmented JSON
                segmented_json = {"words": words}
                with open(os.path.join(segmented_json_path, f"{combined_name}.json"), "w") as f:
                    json.dump(segmented_json, f, indent=4)
