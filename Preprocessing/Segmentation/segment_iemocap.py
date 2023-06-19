# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:51:57 2023

@author: nicol
"""

import json
from pydub import AudioSegment
import os
import pandas as pd

import re

import __libpath
from psychelibrary import psyche_dataset as psd
from psychelibrary import psyche_tools as pt

# A function to get the emotion of a turn
def get_emotion(turn):
    return psyche.df[psyche.df['turn_name'] == turn]['emotion_label'].values[0]

# A function to get the duration of a turn
def get_duration(turn):
    return psyche.df[psyche.df['turn_name'] == turn]['duration'].values[0]


# Function to group words by 'turns' from a given list of words
def group_words_by_turns(words):
    turns = {}
    for i, word in enumerate(words):
        # if 'turn' key is in word, add it to the turn's list of words
        if 'turn' in word:
            if word['turn'] not in turns:
                turns[word['turn']] = []
            turns[word['turn']].append(word)
            
        # if word is '{breath}', add it to the previous turn and next turn
        elif word['word'] == '{breath}':
            
            # go search for the previous normal word's turn
            if i > 0:
                j = 1
                prev_word = words[i-j]
                while prev_word == '{breath}' and i-j >= 0:
                    prev_word = words[i+j]
                    j -= 1
                if 'turn' in prev_word: # if a normal word has been found
                    if prev_word['turn'] not in turns:
                        turns[prev_word['turn']] = []
                    turns[prev_word['turn']].append(word)
                    
                    
            # go search for the next normal word's turn
            if i < len(words)-1:
                j = 1
                next_word = words[i+j]
                while next_word == '{breath}' and i+j < len(words):
                    next_word = words[i+j]
                    j += 1
                if 'turn' in next_word: # if a normal word has been found
                    if next_word['turn'] not in turns:
                        turns[next_word['turn']] = []
                    if 'turn' not in prev_word:
                        turns[next_word['turn']].append(word)
                    elif next_word['turn'] != prev_word['turn']: # it it is the same turn do not append
                        turns[next_word['turn']].append(word)
            
    return turns

# Function to process each turn
def process_turns(turns, turn_names, audio, high_threshold, low_threshold, save_folders):
    
    i = 0
    while i < len(turn_names):
        turn_name = turn_names[i]
        if turn_name not in turns:
            continue
        
        words = turns[turn_name]
        # Create audio segments for each word in turn
        combined = audio[words[0]['start']*1000: words[-1]['end']*1000]

        # If the length of the combined audio is shorter than the lower threshold, try merging with the next turn
        while len(combined) < low_threshold and i < len(turn_names)-1:
            next_turn_name = turn_names[i+1]
            if next_turn_name in turn_names and get_emotion(turn_name) == get_emotion(next_turn_name) and len(combined) + get_duration(next_turn_name)*1000 <= high_threshold:
                i += 1
                words += turns[turn_names[i]]
                combined = audio[words[0]['start']*1000: words[-1]['end']*1000]
            else:
                break

        # If the length of the combined audio is longer than the threshold, split and save it
        if len(combined) > high_threshold: # this is already checked when merging turns, so it will not go here after a merge
            split_and_save(combined, low_threshold, words, turn_name, save_folders)
        else:
            # Else, save it as it is
            save(combined, words, turn_name, turn_name, save_folders)
        i += 1


# Function to split the combined audio into segments of size more than the 'low_threshold' and save each segment
def split_and_save(combined, low_threshold, words, turn, save_folders):
    # Find all the '{breath}' words
    breath_words = [word for word in words if word['word'] == '{breath}']
    segment_start = words[0]['start']
    
    if breath_words:
        # Calculate the mid point of the combined audio
        mid_point = len(combined) // 2

        # Find the '{breath}' closest to the middle
        mid_index = min(range(len(breath_words)), key = lambda index: abs((breath_words[index]['start']-segment_start)*1000 - mid_point))

        # Get the index of the middle '{breath}' word in the original words list
        mid_index_in_words = words.index(breath_words[mid_index])

        # Split the audio and words based on the found '{breath}'
        split1 = combined[:(words[mid_index_in_words]['end']-segment_start)*1000]
        split2 = combined[(words[mid_index_in_words]['start']-segment_start)*1000:]
        words1 = words[:mid_index_in_words+1]
        words2 = words[mid_index_in_words:]

        # Save splits if they meet the lower threshold condition
        if len(split1) >= lower_threshold and len(split2) >= low_threshold:
            save(split1, words1, f"{turn}_0", turn, save_folders)
            save(split2, words2, f"{turn}_1", turn, save_folders)
        else:
            if ASK_CONTINUE:
                input(f"The duration of the split audios ({len(split1)}, {len(split2)}) is smaller than the lower threshold. Press enter to continue anyway...")
            save(split1, words1, f"{turn}_0", turn, save_folders)
            save(split2, words2, f"{turn}_1", turn, save_folders)
    else:
        # If no '{breath}' words are found, simply split at the middle point
        mid_index = len(words) // 2
        split1 = combined[:(words[mid_index]['end']-segment_start)*1000]
        split2 = combined[(words[mid_index]['start']-segment_start)*1000:]
        words1 = words[:mid_index]
        words2 = words[mid_index:]
        
        # Save splits if they meet the lower threshold condition
        if len(split1) >= lower_threshold and len(split2) >= low_threshold:
            save(split1, words1, f"{turn}_0", turn, save_folders)
            save(split2, words2, f"{turn}_1", turn, save_folders)
        else:
            if ASK_CONTINUE:
                input(f"The duration of the split audios ({len(split1)}, {len(split2)}) is smaller than the lower threshold. Press enter to continue anyway...")
            save(split1, words1, f"{turn}_0", turn, save_folders)
            save(split2, words2, f"{turn}_1", turn, save_folders)
        

def save(segment, words, new_turn_name, original_turn, save_folders):
    # Path to the output transcriptions and JSON files
    transcription_path = save_folders[0]
    segmented_json_path = save_folders[1]
    segmented_wav_path = save_folders[3]
    
    segment_dataframe = save_folders[4]
    
    actor_channel = pt.get_iemocap_speaker_channel(original_turn)
    # Save audio
    wav_path = os.path.join(segmented_wav_path, f"{new_turn_name}.wav")
    channels = segment.split_to_mono()
    actor_segment = channels[actor_channel]
    actor_segment.export(wav_path, format="wav")

    # Save transcription
    transcription_path = os.path.join(transcription_path, f"{new_turn_name}.txt")
    transcription = ' '.join([word['word'] for word in words])
    with open(transcription_path, "w") as f:
        f.write(transcription)

    # Save segmented JSON
    alignment_path = os.path.join(segmented_json_path, f"{new_turn_name}.json")
    segmented_json = {"words": words}
    with open(alignment_path, "w") as f:
        json.dump(segmented_json, f, indent=4)

    # Add row to 'segment_data'
    duration = len(segment)
    emotion = get_emotion(original_turn)
    segment_dataframe.loc[len(segment_dataframe)] = [new_turn_name, duration, emotion, wav_path, transcription_path, alignment_path]
    
    if VERBOSE == True:
        print(segment_dataframe.iloc[-1])

#****************************************************************************************
VERBOSE = True
ASK_CONTINUE = False

content = pd.read_pickle("../Save/IEMOCAP/content_processed.pkl")
psyche = psd.PsycheDataset('IEMOCAP', content)
aligner = 'iemocap_default'
psyche.set_aligner(aligner)

# Path to JSON files
ALIGN_DIR = psyche.ALIGN_DIR


breathing_setting_set = 'breath2'
breath_folder = ALIGN_DIR + aligner + '/' + breathing_setting_set
jsons_path = breath_folder + '/json'

save_folders = pt.create_segmenting_save_folders(breath_folder)

json_files = pt.get_json_files(jsons_path)

conv_dict = psyche.audios

# Threshold for splitting longer turns in milliseconds
higher_threshold = 15000
# Threshold for merging shorter turns in milliseconds
lower_threshold = 3000


# DataFrame to store segment data
segment_dataframe = pd.DataFrame(columns=['turn_name', 'duration', 'emotion', 'wav_path', 'transcription_path', 'alignment_path'])
save_folders.append(segment_dataframe)


# For each json file, load words, group them by turns and process each turn
for filename in json_files:
    alignment_dict = pt.read_json(os.path.join(jsons_path, filename))
    words = psyche.get_words(alignment_dict)
    turns_in_file = group_words_by_turns(words)
    conversation_id = os.path.splitext(filename)[0]
    conversation_df = psyche.df.loc[psyche.df['conversation_id'] == conversation_id]
    turn_names = conversation_df.sort_values('start_time')['turn_name'].tolist()
    audio = AudioSegment.from_wav(conv_dict[conversation_id])
    process_turns(turns_in_file, turn_names, audio, higher_threshold, lower_threshold, save_folders)

segment_dataframe.to_pickle("../Save/IEMOCAP/segmented.pkl")