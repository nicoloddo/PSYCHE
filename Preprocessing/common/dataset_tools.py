# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 19:34:46 2023

@author: nicol
"""

# IMPORTS AND GLOBAL VARIABLES
import sys

import os
from os.path import isfile, join
from pathlib import Path

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string

import IPython.display as ipd
import matplotlib.pyplot as plt

from pydub import AudioSegment, silence
from pydub.playback import play

import json

import warnings

warnings.warn("nic_dataset_tools is deprecated.")


AVAILABLE_ALIGNERS = ['gentle_aligner', 'montreal', 'assembly_ai_default', 'whisper', 'iemocap_default']

# GLOBAL VARS TO BE IMPORTED IN OTHER FILES
LIBRARY_DIRECTORY = 'D:/OneDrive - Universiteit Utrecht/Documents/000 - Documenti/PROJECTS/PSYCHE/psychelibrary/'
with open(LIBRARY_DIRECTORY + 'Settings and Secrets/' + 'secrets.json', 'r') as f:
    secrets = json.load(f)
    
BREATHING_SETTINGS_SET = 'breath2'
with open(LIBRARY_DIRECTORY + 'Settings and Secrets/' + 'breath_labeling_settings.json', 'r') as f:
    breath_settings = json.load(f)[BREATHING_SETTINGS_SET]
    

BASE_DIR = secrets["BASE_DIR"]
IBM_URL = secrets["IBM_URL"]
IBM_APIKEY = secrets["IBM_APIKEY"]
ASSAI_APIKEY = secrets["ASSAI_APIKEY"]

RESPIRATION_LENGTH_MIN = float(breath_settings['respiration_length_min']) # in seconds
INTERVAL_DB_MAX = int(breath_settings['interval_db_max']) # in db
INTERVAL_PEAK_DB_MAX = int(breath_settings['interval_peak_db_max']) # in db
RESPIRATION_DB_MAX = int(breath_settings['respiration_db_max']) # in db
BREATH_TOKEN = breath_settings['breath_token']

TIME_FACTOR = 1

ALIGNER = ''
DATASET = 'INTERSPEECH'

if DATASET == 'INTERSPEECH':
    # The defaults are INTERSPEECH
    DEFAULT_dataset_dir_relative = "INTERSPEECH/ComParE2020_Breathing/"
    DEFAULT_transcriptions_dir_relative = 'transcriptions/'
    DEFAULT_alignments_dir_relative = 'alignments'
    DEFAULT_wavs_dir_relative = "normalized_wav/"
    DEFAULT_temp_wavs_dir_relative = "temp_wav/"
    DEFAULT_breath_labels_dir_relative = 'lab/labels.csv' # set to '!nobreathlabels!' if no breath labels are available
    
elif DATASET == 'IEMOCAP':
    DEFAULT_dataset_dir_relative = "IEMOCAP_full_release_withoutVideos.tar/IEMOCAP_full_release_withoutVideos/IEMOCAP_full_release"
    DEFAULT_transcriptions_dir_relative = 'transcriptions/'
    DEFAULT_alignments_dir_relative = 'alignments/'
    DEFAULT_wavs_dir_relative = "wav/"
    DEFAULT_temp_wavs_dir_relative = "temp_wav/"
    DEFAULT_breath_labels_dir_relative = '!nobreathlabels!' # set to '!nobreathlabels!' if no breath labels are available

else:
    sys.exit("You didn't choose a valid Dataset.")



def setup_global_paths(dataset_dir_relative = DEFAULT_dataset_dir_relative, transcriptions_dir_relative = DEFAULT_transcriptions_dir_relative, alignments_dir_relative = DEFAULT_alignments_dir_relative, wavs_dir_relative = DEFAULT_wavs_dir_relative, temp_wavs_dir_relative = DEFAULT_temp_wavs_dir_relative, breath_labels_dir_relative = DEFAULT_breath_labels_dir_relative):
    DATASET_DIR = os.path.join(BASE_DIR, dataset_dir_relative)

    TRANSCR_DIR = os.path.join(DATASET_DIR, transcriptions_dir_relative)
    ALIGN_DIR = os.path.join(DATASET_DIR, alignments_dir_relative)

    WAVS_DIR = os.path.join(DATASET_DIR, wavs_dir_relative)
    TEMP_WAVS_DIR = os.path.join(DATASET_DIR, temp_wavs_dir_relative)
    
    if not breath_labels_dir_relative == '!nobreathlabels!': # If no breath labels are available
        BREATH_DF = pd.read_csv(DATASET_DIR + breath_labels_dir_relative, lineterminator='\n', na_values = '?', dtype={'filename':str, 'timeFrame':np.float64, 'upper_belt':np.float64})
    else:
        BREATH_DF = pd.DataFrame() # Empty DataFrame
        
    WAV_FILENAMES = [f for f in os.listdir(WAVS_DIR) if isfile(join(WAVS_DIR, f))]

    return DATASET_DIR, TRANSCR_DIR, ALIGN_DIR, WAVS_DIR, TEMP_WAVS_DIR, BREATH_DF, WAV_FILENAMES


DATASET_DIR, TRANSCR_DIR, ALIGN_DIR, WAVS_DIR, TEMP_WAVS_DIR, BREATH_DF, WAV_FILENAMES = setup_global_paths() # Here it uses the default ones


#  AUDIO EXPLORATION FUNCTIONS

def get_file_json(filename, json_dir, suffix = ''): # with wav filename
    with open(ALIGN_DIR + json_dir + filename[:-4] + suffix + '.json', 'r') as f:
        data = json.load(f)
    return dict(data)

def get_transcript(filename, specific_dir, suffix = ''): # with wav filename
    with open(TRANSCR_DIR + specific_dir + filename[:-4] + suffix + '.txt', 'r') as f:
        transcript = f.read()
    return transcript

def check_aligner():
    if ALIGNER not in AVAILABLE_ALIGNERS:
        sys.exit("You didn't choose a valid aligner.")

def flatten_whisper_words(audio_json):
    # Collect all words from each segment
    all_words = [word for segment in audio_json['segments'] for word in segment['words']]
    # Add 'words' key with the collected words to the json data
    audio_json['words'] = all_words
    return audio_json

def parse_iemocap_turn_alignment(filepath): # parses a single turn alignment file
    with open(filepath, 'r') as f:
        alignment = f.readlines()
    
    turn_name = os.path.basename(filepath)[:-6]
    
    data = []
    for line in alignment:
        if not line.startswith(" Total") and not line.startswith("\t SFrm"):  # skipping total score and headers
            parts = line.strip().split()
            if len(parts) == 4:  # proper line with data
                entry = {'word': parts[3], 'start': float(parts[0])/100, 'end': float(parts[1])/100, 'score': parts[2], 'turn': turn_name}
                data.append(entry)
    return data

def get_iemocap_turnrow(df, turn_name):
    return df.loc[df['turn_name'] == turn_name].iloc[0]


def get_words_json(audio_json):
    check_aligner()
    if ALIGNER == 'gentle_aligner':
        words = audio_json['words']
    elif ALIGNER == 'montreal':
        words = audio_json['tiers']['words']['entries']
    elif ALIGNER == 'assembly_ai_default':
        words = audio_json['words']
    elif ALIGNER == 'whisper':
        words = flatten_whisper_words(audio_json)['words']
    elif ALIGNER == 'iemocap_default':
        words = audio_json['words']
    return words

def get_info(word):
    check_aligner()
    if ALIGNER == 'gentle_aligner': # (it refers to the sorted one)
        start = word['start']
        end = word['end']
        text = word['word']
    elif ALIGNER == 'montreal':
        start = word[0]
        end = word[1]
        text = word[2]
    elif ALIGNER == 'assembly_ai_default':
        start = word['start']
        end = word['end']
        text = word['text']
    elif ALIGNER == 'whisper':
        start = word['start']
        end = word['end']
        text = word['word']
    elif ALIGNER == 'iemocap_default':
        start = word['start']
        end = word['end']
        text = word['word']
    return start, end, text

def set_word_text(word, text): # overwrites the text field of a word
    check_aligner()
    if ALIGNER == 'gentle_aligner': # (it refers to the sorted one)
        word['word'] = text
    elif ALIGNER == 'montreal':
        word[2] = text
    elif ALIGNER == 'assembly_ai_default':
        word['text'] = text
    elif ALIGNER == 'whisper':
        word['word'] = text
    elif ALIGNER == 'iemocap_default':
        word['word'] = text
    
def seconds_2_index(timestamp, audio_json):
    for i, word in enumerate(get_words_json(audio_json)):
        start, _, _ = get_info(word)
        if start > timestamp:
            if i == 0:
                return i
            else:
                return i-1

def normalize_word(word):
    # Convert to lowercase and remove punctuation
    word = word.lower().strip(string.punctuation)
    # Remove trailing numbers
    if '(' in word:
        word = word.split('(')[0]
    return word

def access_word(i, filename, audio_json, time_factor = TIME_FACTOR):
    words = get_words_json(audio_json)
    # json informations on the word: --------------------
    print(words[i])

    # Breathing signal:----------------------------------
    start, end, text = get_info(words[i])
    
    start = time2seconds(start, time_factor)
    end = time2seconds(end, time_factor)
    
    if not BREATH_DF.empty:
        df = BREATH_DF.loc[BREATH_DF['filename'] == filename]
    
        df = df.loc[df['timeFrame'] >= start]
        df = df.loc[df['timeFrame'] <= end]
    
        print()
        print("Word:", text)
        print()
        print(df)
        plt.figure(figsize=(10,5))
        plt.plot(df['timeFrame'], df['upper_belt'])
        plt.xticks(np.arange(start, end, 0.1), rotation = 45)
        plt.yticks(np.arange(min(BREATH_DF['upper_belt']), max(BREATH_DF['upper_belt']), 0.1), rotation = 45)
        plt.grid(True)
        plt.show()

    # Audio:--------------------------------------------
    start_ms = start * 1000 #Works in milliseconds
    end_ms = end * 1000
    newAudio = AudioSegment.from_wav(WAVS_DIR + filename)
    newAudio = newAudio[start_ms:end_ms]

    newAudio.export(TEMP_WAVS_DIR + 'temp.wav', format="wav")
    ipd.display(ipd.Audio(TEMP_WAVS_DIR + 'temp.wav'))

def access_consecutive_words(i, audio_json):
    words = get_words_json(audio_json)
    
    word1 = words[i]
    if i < len(words)-1:
        word2 = words[i+1]
    else:
        word2 = '<end_token>'
        
    return word1, word2

def time2seconds(timestamp, time_factor):
    # converts into seconds the timestamp by multiplying with the time_factor
    return timestamp * time_factor # time_factor is usually by default 1, supposing therefore we have seconds already

def get_audio_segment(start = 0, end = -1, filename = None):
    start_ms = start * 1000 # Works in milliseconds
    end_ms = end * 1000
    newAudio = AudioSegment.from_wav(WAVS_DIR + filename)

    if end == -1:
        return newAudio
    else:
        newAudio = newAudio[start_ms:end_ms]
        return newAudio  

def get_audio_segment_noroot(start = 0, end = -1, filename = None):
    start_ms = start * 1000 # Works in milliseconds
    end_ms = end * 1000
    newAudio = AudioSegment.from_wav(filename)

    if end == -1:
        return newAudio
    else:
        newAudio = newAudio[start_ms:end_ms]
        return newAudio

def get_iemocap_turn_wav_path(df, turn_name):
    return df[df['turn_name'] == turn_name]['wav_path'].values[0]

def print_iemocap_turn_wav(df, turn_name):
    turn_wav = get_iemocap_turn_wav_path(df, turn_name)
    ipd.display(ipd.Audio(Path(turn_wav)))

def print_audio_segment(start, end, filename):
    newAudio = get_audio_segment(start, end, filename)
    
    newAudio.export(TEMP_WAVS_DIR + 'temp.wav', format="wav")
    ipd.display(ipd.Audio(TEMP_WAVS_DIR + 'temp.wav'))
    print("db:", newAudio.dBFS)
    print("Peak db:", newAudio.max_dBFS)
    print()
        
    return newAudio

def print_spaced_words(word1, word2, filename, time_factor = TIME_FACTOR, images = True, collective_image = True, only_space_audio = False, original_audio = False, printdf = False, printjson = True):
    
    start1, end1, text1 = get_info(word1)
    start2, end2, text2 = get_info(word2)
    
    # convert into seconds (if necessary)
    start1 = time2seconds(start1, time_factor)
    start2 = time2seconds(start2, time_factor)
    end1 = time2seconds(end1, time_factor)
    end2 = time2seconds(end2, time_factor)
    
    text = text1 + ' _ ' + text2
    
    # json informations on the word: --------------------
    if printjson:
        print(word1)
        print()
        print(word2)
        print()

    # Breathing signal:----------------------------------
    if not BREATH_DF.empty:
        df_file = BREATH_DF.loc[BREATH_DF['filename'] == filename]
    
        df = df_file.loc[df_file['timeFrame'] >= start1]
        df = df.loc[df['timeFrame'] <= end2]
    
        if printdf:
            print()
            print(df)
            
    print()
    print("Space start:", end1)
    print("Space end:", start2)
    print("Duration:", start2-end1, "seconds")
    print()
    print("Words:", text, '...')
    
    if not BREATH_DF.empty:
        if images:
            plt.figure(figsize=(10,5))
            plt.plot(df['timeFrame'], df['upper_belt'])
            plt.xticks(np.arange(start1, end2, 0.1), rotation = 45)
            plt.yticks(np.arange(min(BREATH_DF['upper_belt']), max(BREATH_DF['upper_belt']), 0.1), rotation = 45)
            plt.grid(True)
            plt.axvline(x = end1, color = 'b', label = 'axvline - full height')
            plt.axvline(x = start2, color = 'b', label = 'axvline - full height')
            plt.show()
        
        if not collective_image:
            plt.figure(figsize=(120,5))
            plt.plot(df_file['timeFrame'], df_file['upper_belt'])
            plt.xticks(np.arange(min(df_file['timeFrame']), max(df_file['timeFrame'])+1, 0.5), rotation = 45)
            plt.yticks(np.arange(min(BREATH_DF['upper_belt']), max(BREATH_DF['upper_belt']), 0.1), rotation = 45)
            plt.grid(True)
            plt.axvline(x = start1, color = 'r', label = 'axvline - full height')
            plt.axvline(x = end1, color = 'b', label = 'axvline - full height')
            plt.axvline(x = start2, color = 'b', label = 'axvline - full height')
            plt.axvline(x = end2, color = 'r', label = 'axvline - full height')
            plt.show()

    if original_audio:
        ipd.display(ipd.Audio(WAVS_DIR + filename))

    # Audio:--------------------------------------------
    print("Section:")
    print_audio_segment(start1, end2, filename)

    if not only_space_audio:
        # Audio:--------------------------------------------
        print("First word:")
        print_audio_segment(start1, end1, filename)

        # Audio:--------------------------------------------
        print("Second word:")
        print_audio_segment(start2, end2, filename)
    
    # Space Audio:--------------------------------------------
    print("Space in between:")
    print_audio_segment(end1, start2, filename)
            
def search_inconsistencies(filename, audio_json, time_factor = TIME_FACTOR, nonstop = True, incons_threshold = 0.00001): 
    count_inconsistence = 0
    count_wrong_order = 0
    for i in range(len(get_words_json(audio_json))):
        inconstistence = False
        wrong_order = False
        
        if not nonstop:
            ipd.clear_output()

        word1, word2 = access_consecutive_words(i, audio_json)
        if word2 == '<end_token>':
            break

        start1, end1, text1 = get_info(word1)
        start2, end2, text2 = get_info(word2)

        start1 = time2seconds(start1, time_factor)
        start2 = time2seconds(start2, time_factor)
        end1 = time2seconds(end1, time_factor)
        end2 = time2seconds(end2, time_factor)
        
        if start2 < end1 and abs(start2-end1) > incons_threshold and start2 >= start1: # it's an inconsistency if the start of the second is before the end of the first
            inconstistence = True
            count_inconsistence += 1
        elif start2 < start1:
            wrong_order = True
            count_wrong_order += 1
        else:
            continue
        
        text = text1 + ' _ ' + text2
        
        print(word1)
        print()
        print(word2)
        print()
        
        if inconstistence:
            print("i:", i, "- INCONSISTENCY")
        elif wrong_order:
            print("i:", i, "- WRONG ORDER")
        print()
        print("Words:", text, '...')
        print()
        print("Incostistencies:", count_inconsistence)
        print("Wrong orders:", count_wrong_order)

        print("Section:")
        print_audio_segment(start1, start1+10, filename)
        
        if not nonstop:
            stop = input("Continue? [(y)/n] ")
            if stop == "n":
                break
    
    print("************************************************")
    print("Total incostistencies:", count_inconsistence)
    print("Total wrong orders:", count_wrong_order)
            

# BREATHING SEARCH FUNCTIONS
            
def breathing_in_space(word1, word2, filename, time_factor = TIME_FACTOR, length_min = RESPIRATION_LENGTH_MIN, interval_db_max = INTERVAL_DB_MAX, interval_peak_db_max = INTERVAL_PEAK_DB_MAX):
    respiration_length_min = length_min # in seconds
    respiration_db_threshold = interval_db_max # in db
    respiration_peak_db_threshold = interval_peak_db_max

    start1, end1, _ = get_info(word1)
    start2, end2, _ = get_info(word2)
    
    start1 = time2seconds(start1, time_factor)
    start2 = time2seconds(start2, time_factor)
    end1 = time2seconds(end1, time_factor)
    end2 = time2seconds(end2, time_factor)
    
    # get audio between words ---------------------------
    start_ms = end1 * 1000 #Works in milliseconds
    end_ms = start2 * 1000
    breathAudio = AudioSegment.from_wav(WAVS_DIR + filename)
    breathAudio = breathAudio[start_ms:end_ms]
    
    # check if we should return True: is it (not) breathing? --
    if start2-end1 < respiration_length_min:
        return False
    elif breathAudio.dBFS > respiration_db_threshold:
        return False
    elif breathAudio.max_dBFS > respiration_peak_db_threshold:
        return False
    else:
        return True
    
def breathing_in_word(word, filename, time_factor = TIME_FACTOR, length_min = RESPIRATION_LENGTH_MIN, interval_db_max = INTERVAL_DB_MAX, interval_peak_db_max = INTERVAL_PEAK_DB_MAX):
    respiration_length_min = length_min # in seconds
    respiration_db_threshold = interval_db_max # in db
    respiration_peak_db_threshold = interval_peak_db_max
    
    start, end, _ = get_info(word)
    start = time2seconds(start, time_factor)
    end = time2seconds(end, time_factor)
    
    # get audio between words ---------------------------
    start_ms = start * 1000 #Works in milliseconds
    end_ms = end * 1000
    breathAudio = AudioSegment.from_wav(WAVS_DIR + filename)
    breathAudio = breathAudio[start_ms:end_ms]
    
    # check if we should return True: is it (not) breathing? --
    if end-start < respiration_length_min:
        return False
    elif breathAudio.dBFS > respiration_db_threshold:
        return False
    elif breathAudio.max_dBFS > respiration_peak_db_threshold:
        return False
    else:
        return True
    
def search_breath_analysis(filename, audio_json, time_factor = TIME_FACTOR, interval_db_max = INTERVAL_DB_MAX, respiration_db_max = RESPIRATION_DB_MAX, interval_peak_db_max = INTERVAL_PEAK_DB_MAX, respiration_length_min = RESPIRATION_LENGTH_MIN, nonstop=False, printonlyfinal=False, printjson = False, askcontinue=False):
    if nonstop:
        if not BREATH_DF.empty:
            df_file = BREATH_DF.loc[BREATH_DF['filename'] == filename]
            plt.figure(figsize=(120,5))
            plt.plot(df_file['timeFrame'], df_file['upper_belt'])
            plt.xticks(np.arange(min(df_file['timeFrame']), max(df_file['timeFrame'])+1, 0.5), rotation = 45)
            plt.yticks(np.arange(min(BREATH_DF['upper_belt']), max(BREATH_DF['upper_belt']), 0.1), rotation = 45)
            plt.grid(True)
    else:
        printonlyfinal = False
    
    count_potential_breath = 0
    count_breath_sections = 0
    
    for i in range(len(get_words_json(audio_json))):
        if not nonstop:
            ipd.clear_output()

        word1, word2 = access_consecutive_words(i, audio_json)
        if word2 == '<end_token>':
            break

        if not breathing_in_space(word1, word2, filename, time_factor, respiration_length_min, interval_db_max, interval_peak_db_max):
            continue
        
        start1, end1, text1 = get_info(word1)
        start2, end2, text2 = get_info(word2)

        start1 = time2seconds(start1, time_factor)
        start2 = time2seconds(start2, time_factor)
        end1 = time2seconds(end1, time_factor)
        end2 = time2seconds(end2, time_factor)
        
        
        count_potential_breath += 1
        
        if not printonlyfinal:
            print("i:", i)
            print("Potential breath spaces:", count_potential_breath)
            print_spaced_words(word1, word2, filename, time_factor = time_factor, images = not nonstop, collective_image = nonstop, only_space_audio= nonstop, printjson = printjson)
            print()
            
        # Find breath inside the space
        space_audio = get_audio_segment(end1, start2, filename)
        breath_section = silence.detect_silence(space_audio, min_silence_len=int(respiration_length_min*1000), silence_thresh=respiration_db_max, seek_step = 1)
        
        
        if len(breath_section) > 0: # if breath has been detected
            count_breath_sections += len(breath_section)
            if nonstop: # place red bar for the audio section
                if not BREATH_DF.empty:
                    plt.axvline(x = start1, color = 'r', label = 'axvline - full height')
                    plt.axvline(x = end2, color = 'r', label = 'axvline - full height')
            
            if len(breath_section) == 1 and round((breath_section[0][1] - breath_section[0][0])/1000, 3) == round(start2-end1, 3):
                if not printonlyfinal:
                    print("THE WHOLE SPACE IS A BREATH.")                
                if nonstop: # place blue bars for the breath sections
                    if not BREATH_DF.empty:
                        plt.axvline(x = end1, color = 'b', label = 'axvline - full height')
                        plt.axvline(x = start2, color = 'b', label = 'axvline - full height')
                    
            else:
                if not printonlyfinal:
                    print("BREATH SECTIONS:")
                for breath in breath_section:
                    start_timestamp = end1 + breath[0]/1000
                    end_timestamp = end1 + breath[1]/1000
                    
                    if not printonlyfinal:
                        print_audio_segment(start_timestamp, end_timestamp, filename)
                        print(breath)
                        print("Duration:", end_timestamp - start_timestamp, "seconds")
                        print()
                    
                    if nonstop: # place blue bars for the breath sections
                        if not BREATH_DF.empty:
                            plt.axvline(x = start_timestamp, color = 'b', label = 'axvline - full height')
                            plt.axvline(x = end_timestamp, color = 'b', label = 'axvline - full height')
        
        else: # there are no breaths
            if not printonlyfinal:
                print("NO BREATH FOUND IN THIS SPACE SECTION.")
                if nonstop: # place red bar for the audio section even though there is no breath
                    if not BREATH_DF.empty:
                        plt.axvline(x = start1, color = 'r', label = 'axvline - full height')
                        plt.axvline(x = end2, color = 'r', label = 'axvline - full height')
        
        if not printonlyfinal:
            print("Total breath sections:", count_breath_sections)      
            print("************************************************")
        
        if not nonstop:
            if askcontinue:
                stop = input("Continue? [(y)/n] ")
                if stop == "n":
                    break
            else:
                break
        
    if nonstop:
        if printonlyfinal:
            print("Final total breath sections:", count_breath_sections)      
            print("************************************************")
        if not BREATH_DF.empty:
            plt.show()

def argmax_max(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])
def argmin_min(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])

def breath_stats(alignment_dir, time_factor = TIME_FACTOR, interval_db_max = INTERVAL_DB_MAX, respiration_db_max = RESPIRATION_DB_MAX, interval_peak_db_max = INTERVAL_PEAK_DB_MAX, respiration_length_min = RESPIRATION_LENGTH_MIN, singleprint = True): 
    # search for breaths across an alignment directory and gives statistics about them
    # alignment_dir = the general directory of the alignment, not the json one.
    
    full_json_path = ALIGN_DIR + alignment_dir + 'json/'
    json_names = [f for f in os.listdir(full_json_path) if isfile(join(full_json_path, f))]
    
    breath_stats = {'counts':[], 'averages':[], 'averages_nobreath':[], 'maxs':[], 'mins':[], 'maxs_nobreath':[], 'mins_nobreath':[], 'filestats':{}}
    for i, filename in enumerate(json_names): 
        print(i+1, '/', len(json_names))
        print("Searching in:", filename)
        wav_filename = filename[:-5] + '.wav'
        
        with open(full_json_path + filename, 'r') as f:
            data = json.load(f)
            alignment_json = dict(data)
        words = get_words_json(alignment_json)
        first_word_start, _, _ = get_info(words[0])
        
        count_potential_breath = 0
        count_breath_sections = 0
        
        breath_durations = list()
        nobreath_durations = list()
        start_nobreath = first_word_start
        for i in range(len(words)):
    
            word1, word2 = access_consecutive_words(i, alignment_json)
            
            start1, end1, text1 = get_info(word1)
            
            if word2 == '<end_token>':
                break
            start2, end2, text2 = get_info(word2)
            
            # if there is no breathing in the space:
            if not breathing_in_space(word1, word2, wav_filename, time_factor, respiration_length_min, interval_db_max, interval_peak_db_max): # using the default threshold of ndt
                continue
            
            count_potential_breath += 1
    
            start1 = time2seconds(start1, time_factor)
            start2 = time2seconds(start2, time_factor)
            end1 = time2seconds(end1, time_factor)
            end2 = time2seconds(end2, time_factor)        
            
            # Find breath inside the space
            space_audio = get_audio_segment(end1, start2, wav_filename)
            breath_section = silence.detect_silence(space_audio, min_silence_len=int(respiration_length_min*1000), silence_thresh=respiration_db_max, seek_step = 1)
            
            count_breath_sections += len(breath_section)
            for i, breath in enumerate(breath_section):
                start_timestamp = end1 + breath[0]/1000
                end_timestamp = end1 + breath[1]/1000
                duration = end_timestamp - start_timestamp # in seconds
                breath_durations.append(round(duration, 2))
                
                if i == 0: # from no_breath start to the start of the first breath in this section is the no breath duration
                    nobreath_durations.append(round(start_timestamp - start_nobreath, 2))
                if i == len(breath_section) - 1: # we reset the start of speech to the start of the second word
                    start_nobreath = start2
        
        average = round(sum(breath_durations)/len(breath_durations), 2)
        average_nobreath = round(sum(nobreath_durations)/len(nobreath_durations), 2)
        max_duration = max(breath_durations)
        min_duration = min(breath_durations)
        max_duration_nobreath = max(nobreath_durations)
        min_duration_nobreath = min(nobreath_durations)
        
        breath_stats['filestats'][filename] = breath_durations
        breath_stats['counts'].append(count_breath_sections)
        breath_stats['averages'].append(average)
        breath_stats['averages_nobreath'].append(average_nobreath)
        breath_stats['maxs'].append(max_duration)
        breath_stats['mins'].append(min_duration)
        breath_stats['maxs_nobreath'].append(max_duration_nobreath)
        breath_stats['mins_nobreath'].append(min_duration_nobreath)
        
        if singleprint:
            ipd.clear_output()
        print(filename)
        print("Potential breath sections:", count_potential_breath)
        print("Actual breath sections:", count_breath_sections)
        print("Average breath duration:", average)
        print("Average duration till getting a breath:", average_nobreath)
        print("Max breath duration:", max_duration)
        print("Min breath duration:", min_duration)
        print("Max duration till getting a breath:", max_duration_nobreath)
        print("Min duration till getting a breath:", min_duration_nobreath)
                
        print("************************************************")
        print()
    
    count_breath_average = round(sum(breath_stats['counts']) / len(breath_stats['counts']), 2)
    max_breaths_i, max_breaths = argmax_max(breath_stats['counts'])
    min_breaths_i, min_breaths = argmin_min(breath_stats['counts'])
    max_duration_i, max_duration_total = argmax_max(breath_stats['maxs'])
    min_duration_i, min_duration_total = argmin_min(breath_stats['mins'])
    total_duration_average = round(sum(breath_stats['averages']) / len(breath_stats['averages']), 2)
    total_nobreath_duration_average = round(sum(breath_stats['averages_nobreath']) / len(breath_stats['averages_nobreath']), 2)
    
    if singleprint:
        ipd.clear_output()
    
    print("FINAL STATS:")
    print("Average number of breaths:", count_breath_average)
    print("Max breaths:", max_breaths, "-", json_names[max_breaths_i])
    print("Min breaths:", min_breaths, "-", json_names[min_breaths_i])
    print("Average breath duration:", total_duration_average)
    print("Average duration till getting a breath:", total_nobreath_duration_average)
    print("Max breath duration:", max_duration_total, "-", json_names[max_duration_i])
    print("Min breath duration:", min_duration_total, "-", json_names[min_duration_i])
    print("Max duration till getting a breath:", max(breath_stats['maxs_nobreath']))
    print("Min duration till getting a breath:", min(breath_stats['mins_nobreath']))
    
    return breath_stats

def create_save_folder(path):
    exists = os.path.exists(path)
    if not exists:
       # Create the save directory
       os.makedirs(path)
       
def segment_files(breath_alignment_dir, json_filename, max_segment_length, min_segment_length, print_segments = True, save_segments = False):    
    create_save_folder(TRANSCR_DIR + breath_alignment_dir + 'segmented_alignment/json/')
    create_save_folder(TRANSCR_DIR + breath_alignment_dir + 'segmented_alignment/')
    create_save_folder(TRANSCR_DIR + breath_alignment_dir + 'segmented_tokens/')
    create_save_folder(TRANSCR_DIR + breath_alignment_dir + 'segmented_wav/')
    
    full_json_path = TRANSCR_DIR + breath_alignment_dir + 'json/'
    with open(full_json_path + json_filename, 'r') as f:
        data = json.load(f)
        alignment_json = dict(data)
    words = get_words_json(alignment_json)        
    
    i = 0
    start, end, text = get_info(words[i]) # the initialization is the start of the first word or breath in the alignment
    segment_start_index = i # the index of the first word of segment
    segment_start = start
    
    for i in range(1, len(words)):
        start, end, text = get_info(words[i]) # select next word
        
        if end - segment_start > max_segment_length: # if the segment has reached the max length without finding a breath
            print_save_segments_by_index(segment_start_index, i-1, words, json_filename, print_segments, save_segments, breath_alignment_dir) # save stopping at the previous word
            segment_start_index = i # set the current word (breath) as start index of next segment
            segment_start = start
            
        elif text == BREATH_TOKEN: # if it is a breath token
            if end - segment_start < min_segment_length: # if the segment is too little, we go search the next breath
                continue
            else: # the segment is not too little and there is a breath token
                print_save_segments_by_index(segment_start_index, i, words, json_filename, print_segments, save_segments, breath_alignment_dir) # save
                segment_start_index = i # set the current word (breath) as start index of next segment
                segment_start = start            
            
    
def print_save_segments_by_index(start_index, end_index, words, json_filename, print_segments = True, save_segments = False, alignment_dir = '', token_segment = True):
    save_json_dir = TRANSCR_DIR + alignment_dir + 'segmented_alignment/json/' + json_filename[:-5] + '_' + str(start_index) + '-' + str(end_index) + '.txt'
    save_transcr_dir = TRANSCR_DIR + alignment_dir + 'segmented_alignment/' + json_filename[:-5] + '_' + str(start_index) + '-' + str(end_index) + '.txt'
    save_token_transcr_dir = TRANSCR_DIR + alignment_dir + 'segmented_tokens/' + json_filename[:-5] + '_' + str(start_index) + '-' + str(end_index) + '.txt'
    save_wav_dir = TRANSCR_DIR + alignment_dir + 'segmented_wav/' + json_filename[:-5] + '_' + str(start_index) + '-' + str(end_index) + '.wav'
    
    wav_filename = json_filename[:-5] + '.wav'
    
    segm_transcr = ''
    segm_token_transcr = ''
    segm_alignment = {'words':list()}
    start_segm, end_segm, text = get_info(words[start_index]) # first word
    for i in range(start_index, end_index + 1): # without the +1 end_index would be skipped
        segm_alignment['words'].append(words[i])
        start, end, text = get_info(words[i])
        if 'converted' not in words[i]:
            sys.exit("Error: you did not run the token converter.")
        segm_transcr += words[i]['converted'] + ' '
        if token_segment:
            segm_token_transcr += words[i]['token'] + ' '
        
    if print_segments:
        print(segm_transcr)
        print("Duration:", round(end - start_segm, 2))
        print()
        print_audio_segment(start_segm, end, wav_filename)
    
    if save_segments:
        with open(save_json_dir, 'w') as f:
            json.dump(segm_alignment, f, indent = 1) # save the alignment
        with open(save_transcr_dir, 'w') as f:
            f.write(segm_transcr[:-1]) # save the transcription
        if token_segment:
            with open(save_token_transcr_dir, 'w') as f:
                f.write(segm_token_transcr[:-1]) # save the token transcription
        segmAudio = get_audio_segment(start_segm, end, wav_filename)
        segmAudio.export(save_wav_dir, format="wav") # save the audio