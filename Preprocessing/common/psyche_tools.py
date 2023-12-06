# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:27:01 2023

@author: nicol
"""
import os

import json
import string
import re

from pydub import AudioSegment
import IPython.display as ipd

emotion_dict = {
    'neu'  :  0,
    'fru'  :  1,
    'sad'  :  2,
    'ang'  :  3,
    'hap'  :  4,
    'exc'  :  5,
    'sur'  :  6,
    'fea'  :  7,
    'dis'  :  8,
    # Include all possible emotions in your dataset.
}

disfluency_dict = {
    "ah" : '<disfl.ah>', 
    "ahh" : '<disfl.ahh>',
    "eh" : '<disfl.eh>',
    "eeh" : '<disfl.eeh>',
    "ehh" : '<disfl.eh>',
    "um" : '<disfl.um>',
    "hum" : '<disfl.um>',
    "mm" : '<disfl.um>', 
    "hmm" : '<disfl.um>',
    "uhm" : '<disfl.um>',
    "uh" : '<disfl.uh>',
    "uhh" : '<disfl.uh>',
    "oh" : '<disfl.oh>', 
    "ooh" : '<disfl.ooh>',
    "oo" : '<disfl.ooh>',
    "ohh" : '<disfl.ooh>',
    "er" : '<disfl.er>',  
    "huh" : '<disfl.huh>',
    "mhm" : '<disfl.mhm>',
    "mmhmm" : '<disfl.mhm>',
    "uhuh" : '<disfl.mhm>',
    "uh-huh" : '<disfl.mhm>',
    "shh" : '<disfl.shh>',
    "erm" : '<disfl.erm>', 
    "ehm" : '<disfl.ehm>', 
    "oops" : '<disfl.oops>', 
    "phew" : '<disfl.phew>',
    "psst" : '<disfl.psst>',
    "yoo-hoo" : '<disfl.yoo-hoo>', 
    "yikes" : '<disfl.yikes>',
    "ouch" : '<disfl.ouch>',
    "tsk" : '<disfl.tsk>',
    "tsk-tsk" : '<disfl.tsk>', 
    "uh-oh" : '<disfl.uh-oh>', 
    "ahem" : '<disfl.ahem>',
    "argh" : '<disfl.argh>', 
    "urgh" : '<disfl.urgh>',
}

#******************************* CONDITIONALS
def set_word_text(aligner, word, text): # overwrites the text field of a word
    if aligner == 'gentle_aligner': # (it refers to the sorted one)
        word['word'] = text
    elif aligner == 'montreal':
        word[2] = text
    elif aligner == 'assembly_ai_default':
        word['text'] = text
    elif aligner == 'whisper':
        word['word'] = text
    elif aligner == 'iemocap_default':
        word['word'] = text

def get_time_factor(aligner):     
    if aligner == 'gentle_aligner':
        time_factor = 1
    elif aligner == 'montreal':
        time_factor = 1
    elif aligner == 'assembly_ai_default':
        time_factor = 1
    elif aligner == 'whisper':
        time_factor = 1
    elif aligner == 'iemocap_default':
        time_factor = 1
    return time_factor

def get_words(aligner, dict_alignment):
    if aligner == 'gentle_aligner':
        words = dict_alignment['words']
    elif aligner == 'montreal':
        words = dict_alignment['tiers']['words']['entries']
    elif aligner == 'assembly_ai_default':
        words = dict_alignment['words']
    elif aligner == 'whisper':
        words = flatten_whisper_words(dict_alignment)['words']
    elif aligner == 'iemocap_default':
        words = dict_alignment['words']
    return words

def get_info(aligner, word):
    if aligner == 'gentle_aligner': # (it refers to the sorted one)
        start = word['start']
        end = word['end']
        text = word['word']
    elif aligner == 'montreal':
        start = word[0]
        end = word[1]
        text = word[2]
    elif aligner == 'assembly_ai_default':
        start = word['start']
        end = word['end']
        text = word['text']
    elif aligner == 'whisper':
        start = word['start']
        end = word['end']
        text = word['word']
    elif aligner == 'iemocap_default':
        start = word['start']
        end = word['end']
        text = word['word']
    return start, end, text

def get_iemocap_speaker_channel(turn_name):
    session, leading_actor_gender, convo_type, convo_num, actor_speaking_gender, turn_num = re.match(r'(\S{5})([FM])_(\S+?)(\d{2}(?:_\d)?\w*)_([FM])(\d{2})', turn_name).groups()
    if session == 'Ses01': # Session 1 is the only one in which the channel of the leading actor is the left one.
        leading_actor_channel = '0' # left audio channel
        company_actor_channel = '1' # right audio channel
    else:
        leading_actor_channel = '1' # left audio channel
        company_actor_channel = '0' # right audio channel
        
    if actor_speaking_gender == leading_actor_gender:
        return int(leading_actor_channel)
    else:
        return int(company_actor_channel)
        
#******************************* GENERAL TOOLS
def get_json_files(json_path): # Function to return all json files in the given directory
    return [file for file in os.listdir(json_path) if file.endswith('.json')]

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        alignment_dict = dict(data)
    return alignment_dict

def get_json(path): # with wav filename
    with open(path, 'r') as f:
        return json.load(f)
    
def get_transcript(path): # with wav filename
    with open(path, 'r') as f:
        transcript = f.read()
    return transcript

def get_audio_segment(start = 0, end = -1, filename = None):
    start_ms = start * 1000 # Works in milliseconds
    end_ms = end * 1000
    newAudio = AudioSegment.from_wav(filename)

    if end == -1:
        return newAudio
    else:
        newAudio = newAudio[start_ms:end_ms]
        return newAudio
    
def print_audio_segment(start, end, conversation_id, audio_path, temp_wavs_dir):
    newAudio = get_audio_segment(start, end, audio_path)
    
    newAudio.export(temp_wavs_dir + 'temp.wav', format="wav")
    ipd.display(ipd.Audio(temp_wavs_dir + 'temp.wav'))
    print("db:", newAudio.dBFS)
    print("Peak db:", newAudio.max_dBFS)
    print()
        
    return newAudio
    
def access_consecutive_words(i, words):
    word1 = words[i]
    if i < len(words)-1:
        word2 = words[i+1]
    else:
        word2 = '<end_token>'
        
    return word1, word2

def argmax_max(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])
def argmin_min(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])
    
def normalize_word(word):
    # Convert to lowercase and remove punctuation
    word = word.lower().strip(string.punctuation)
    # Remove trailing numbers
    if '(' in word:
        word = word.split('(')[0]
    return word

def time2seconds(timestamp, time_factor):
    # converts into seconds the timestamp by multiplying with the time_factor
    return timestamp * time_factor # time_factor is usually by default 1, supposing therefore we have seconds already

def create_save_folder(path):
    exists = os.path.exists(path)
    if not exists:
       # Create the save directory
       os.makedirs(path)

def create_segmenting_save_folders(breath_folder):
    save_folders = [
        breath_folder + '/segmented_alignment/',
        breath_folder + '/segmented_alignment/json/',
        breath_folder + '/segmented_tokens/',
        breath_folder + '/segmented_wav/',
        ]
    
    for folder in save_folders:
        create_save_folder(folder)
    
    return save_folders
    
def seconds_2_index(timestamp, words, aligner):
    for i, word in enumerate(words):
        start, _, _ = get_info(aligner, word)
        if start > timestamp:
            if i == 0:
                return i
            else:
                return i-1

#******************************* WHISPER TOOLS
def flatten_whisper_words(json):
    # Collect all words from each segment
    all_words = [word for segment in json['segments'] for word in segment['words']]
    # Add 'words' key with the collected words to the json data
    json['words'] = all_words
    return json