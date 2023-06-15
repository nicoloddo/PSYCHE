# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:53:42 2023

@author: nicol
"""
import pandas as pd
from pydub import AudioSegment
import noisereduce as nr
import numpy as np
from scipy.io.wavfile import read, write

import IPython.display as ipd

import sys
import os
import warnings
import json

from psychelibrary import psyche_tools as pt
from psychelibrary import breath_handler as bh

LIBRARY_DIRECTORY = 'D:/OneDrive - Universiteit Utrecht/Documents/000 - Documenti/PROJECTS/PSYCHE/psychelibrary/'
SETTINGS_DIRECTORY = LIBRARY_DIRECTORY + 'Settings and Secrets/'
SECRETS_DIRECTORY = LIBRARY_DIRECTORY + 'Settings and Secrets/'
with open(SECRETS_DIRECTORY + 'secrets.json', 'r') as f:
    secrets = json.load(f)
    BASE_DIR = secrets["BASE_DIR"]
    IBM_URL = secrets["IBM_URL"]
    IBM_APIKEY = secrets["IBM_APIKEY"]
    ASSAI_APIKEY = secrets["ASSAI_APIKEY"]

MANDATORY_COLUMNS = ['conversation_id', 'start_time', 'end_time', 'duration', 'conversation_wav_path']
AVAILABLE_ALIGNERS = ['gentle_aligner', 'montreal', 'assembly_ai_default', 'whisper', 'iemocap_default']

class PsycheDataset():
    '''
    Wrapper around Datasets to be used in the PSYCHE project.
    The Dataset needs to be built having specific columns when parsed to achieve efficient generalizability.
    If the wavs encountered preprocessing, save the new paths in the wav_path column.
    If you want to keep the path of unprocessed audios, just create a new column named unprocessed_wav.
    Consider wav as a generalization of audio formats.
    '''
    
    def __init__(self, dataset_name, dataframe, breath_settings_set = 'breath2', breath_df = None):
        for column in MANDATORY_COLUMNS:
            if column not in dataframe.columns:
                sys.exit("Dataframe is missing a mandatory column:", column)
                
        self.df = dataframe
        
        self.audios = dict(zip(self.df['conversation_id'], self.df['conversation_wav_path']))
        
        self.dataset_name = dataset_name
        
        self.aligner = None
        self.alignments = {}
        
        self.set_dataset_paths()
        self.bh = bh.BreathHandler(breath_settings_set, SETTINGS_DIRECTORY, self.audios, self.TEMP_WAVS_DIR, breath_df)
    
    # CHECKERS
    def check_aligner(self):
        if self.aligner not in AVAILABLE_ALIGNERS:
            sys.exit("You didn't choose a valid aligner.")
    
    # SETTERS
    def set_aligner(self, aligner):
        self.aligner = aligner
        self.check_aligner()
        self.time_factor = pt.get_time_factor(self.aligner)
        
    def set_dataset_paths(self):
        if self.dataset_name == 'INTERSPEECH':
            # The defaults are INTERSPEECH
            dataset_dir_relative = "INTERSPEECH/ComParE2020_Breathing/"
            transcriptions_dir_relative = 'transcriptions/'
            alignments_dir_relative = 'alignments'
            temp_wavs_dir_relative = "temp_wav/"
            
        elif self.dataset_name == 'IEMOCAP':
            dataset_dir_relative = "IEMOCAP_full_release_withoutVideos.tar/IEMOCAP_full_release_withoutVideos/IEMOCAP_full_release"
            transcriptions_dir_relative = 'transcriptions/'
            alignments_dir_relative = 'alignments/'
            temp_wavs_dir_relative = "temp_wav/"

        else:
            sys.exit("You didn't choose a valid Dataset.")
            
        self.DATASET_DIR = os.path.join(BASE_DIR, dataset_dir_relative)
        self.TRANSCR_DIR = os.path.join(self.DATASET_DIR, transcriptions_dir_relative)
        self.ALIGN_DIR = os.path.join(self.DATASET_DIR, alignments_dir_relative)
        self.TEMP_WAVS_DIR = os.path.join(self.DATASET_DIR, temp_wavs_dir_relative)
        
            
    def set_word_text(self, word, text): # overwrites the text field of a word
        self.check_aligner()
        pt.set_word_text(self.aligner, word, text)
    
    # GETTERS
    def get_row(self, column, value):
        '''
        This function before was IEMOCAP specific and called get_iemocap_turnrow.
        If multiple rows satisfy the condition, it returns the first.

        Parameters
        ----------
        column : the column criteria of selection
        column : the value of the column of the wanted row

        Returns
        -------
        The row

        '''
        return self.df.loc[self.df[value] == column].iloc[0]
    
    def get_words(self, alignment_dict):
        self.check_aligner()
        return pt.get_words(self.aligner, alignment_dict)

    def get_info(self, word):
        self.check_aligner()
        return pt.get_info(self.aligner, word)
    
    def load_alignments(self, path = '../Save/alignments.json'):
        with open(path, 'r') as f:
            self.alignments = json.load(f)
    
    #*********************************************  ALIGNMENTS PARSING
    def parse_alignment_file(self, filename, alignment_id):
        
        json_path = self.ALIGN_DIR + alignment_id + '/' + 'json/' + filename
        
        if not os.path.isfile(json_path):
            warnings.warn("\n[In parse_alignment_file()]: Some alignment files are not found.")
            return None
            
        alignment_dict = pt.read_json(json_path)
        words_alignment = self.get_words(alignment_dict)
        
        return words_alignment
    
    def parse_alignment_turn_iemocap(self, row):
        '''
        IEMOCAP default aligner have specifically structured files as alignments.
        This function can be used if the dataset is IEMOCAP, because in the IEMOCAP we have one turn per row instead of one conversation per row
        
        Parameters
        ----------
        row : a row of the dataframe

        Returns
        -------
        data : the parsed alignment

        '''
        
        filepath = row['alignment_path']
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
    
    def parse_alignment_iemocap_dialog(self, dialog_subset):
        '''
        Parses alignment of dialogs in IEMOCAP dataset

        Parameters
        ----------
        dialog_subset : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if self.dataset_name != 'IEMOCAP':
            sys.exit("Dataset has to be IEMOCAP for this function.")
        
        dialog_words = dialog_subset.apply(self.parse_alignment_turn_iemocap, axis=1)
        
        flat_dialog_words = [item for sublist in dialog_words.tolist() for item in sublist]
        
        # Convert dataframe to dictionary for faster lookup
        dialog_dict = dialog_subset.set_index('turn_name').to_dict('index')
        
        for turn_word in flat_dialog_words:
            # Get start_time_of_turn and actor from the dictionary instead of dataframe
            turn_info = dialog_dict[turn_word['turn']]
            start_time_of_turn = float(turn_info['start_time'])
            actor = turn_info['actor_speaking_id']
        
            turn_word['start'] = float(turn_word['start']) + start_time_of_turn
            turn_word['end'] = float(turn_word['end']) + start_time_of_turn
            turn_word['actor'] = actor
            
        return sorted(flat_dialog_words, key=lambda d: d['start'])
        
    
    def parse_alignment_audio(self, row, alignment_id):
        '''
        Wraps around the different structures of datasets to load the alignment.
        Parses alignment of an audio row in the dataset: usually it's one audio per row. 
        In case of IEMOCAP there are sentences per row, therefore it merges them and parses the one of one full dialog.
        Remember that it will repeat this for every turn if you pass the full iemocap dataset.

        Returns
        -------
        None.

        '''
        
        if self.dataset_name == 'IEMOCAP': #--------------------------------------
            convo_id = row['conversation_id']
            if convo_id in self.alignments[alignment_id]['retrieved'] or convo_id in self.alignments[alignment_id]['failed']: # Because IEMOCAP goes turn by row, not conversation by row
                return
            else:
                print("Retrieving", convo_id)
            
            # ********************************************************************
            if alignment_id == 'iemocap_default':
                dialog_subset = self.df.loc[self.df['conversation_id'] == convo_id]
                
                missing_alignments_turns = dialog_subset[dialog_subset['alignment_path'].apply(lambda x: not os.path.isfile(x))]['turn_name'].tolist()
                if len(missing_alignments_turns) != 0: # the conversation has missing turns alignments
                    self.alignments[alignment_id]['failed'].append(convo_id)
                    self.alignments[alignment_id]['fail_reasons'].append({'id' : convo_id, 'reason' : missing_alignments_turns})
                    convo_alignment = None
                else:
                    convo_alignment = self.parse_alignment_iemocap_dialog(dialog_subset)
                
                if convo_alignment is not None:
                    self.alignments[alignment_id]['retrieved'].append(convo_id)
                    self.alignments[alignment_id]['audios'][convo_id] = convo_alignment
            
            # ********************************************************************
            elif alignment_id == 'whisper':                
                filename = row['conversation_id'] + '.json'
                convo_alignment = self.parse_alignment_file(filename, alignment_id)
                if convo_alignment is not None:
                    self.alignments[alignment_id]['retrieved'].append(convo_id)
                    self.alignments[alignment_id]['audios'][convo_id] = convo_alignment
                else:
                    self.alignments[alignment_id]['failed'].append(convo_id)
                    self.alignments[alignment_id]['fail_reasons'].append({'id' : convo_id, 'reason' : 'missing_file'})
            
            else:
                sys.exit("This alignment is invalid or not available for this dataset.")
    
    def parse_alignments(self, alignment_id):
        '''
        Applies the alignment parsing functions to each row of self.dataframe

        Parameters
        ----------
        alignment_id : string
            The alignment identifier, like 'whisper'.

        Returns
        -------
        None.

        '''
        self.alignments[alignment_id] = {'retrieved' : [], 'failed' : [], 'fail_reasons' : [], 'audios' : {}}
        self.df.apply(self.parse_alignment_audio, args=[alignment_id], axis=1)
    
    #*********************************************  PREPROCESSING
    def separate_channels_row(self, row, temp_folder = '../Temp/'): #!!! Still testing
        '''
        Separate channels of each row and saves them. 
        In IEMOCAP the speakers are roughly separated in channels.
        
        Parameters
        ----------
        row : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Load the audio file
        audio = AudioSegment.from_wav(row['conversation_wav_path'])

        # Cut the audio to the required duration
        start_time_ms = row['start_time'] * 1000  # start_time should be in seconds
        end_time_ms = row['end_time'] * 1000  # end_time should be in seconds
        audio = audio[start_time_ms:end_time_ms]

        # Split stereo to mono
        channels = audio.split_to_mono()

        # Denoise each channel
        for i, channel in enumerate(channels):
            # Export channel to a temporary file
            channel.export(temp_folder + "temp" + str(i+1) + ".wav", format="wav")
            
        # Read wav file to array
        rate1, data1 = read(temp_folder + "temp1.wav")
        rate2, data2 = read(temp_folder + "temp2.wav")
        
        return np.array(data1, dtype=np.int16), rate1, np.array(data2, dtype=np.int16), rate2
        
    def separate_speaker_channels(self):
        self.df.apply(self.separate_channels_row, axis=1)
        
    def diarize(self, aligner):
        pass
    
    
    #*********************************************  SEGMENTATION
    def segment_by_breath(self, alignment_id, alignments_relative_path, max_segment_length, min_segment_length, print_segments = True, save_segments = False):    
        
        save_folders = [
            self.ALIGN_DIR + alignment_id + '/breath/segmented_alignment/',
            self.ALIGN_DIR + alignment_id + '/breath/segmented_alignment/json/',
            self.ALIGN_DIR + alignment_id + '/breath/segmented_tokens/',
            self.ALIGN_DIR + alignment_id + '/breath/segmented_wav/',
            ]
        
        for folder in save_folders:
            pt.create_save_folder(folder)
        
        jsons_path = self.ALIGN_DIR + alignments_relative_path
        json_names = [f for f in os.listdir(jsons_path) if os.path.isfile(os.path.join(jsons_path, f))]
        
        for filename in json_names:
            conversation_id = filename[:-5]
            
            json_path = self.ALIGN_DIR + alignment_id + '/breath/json/'
            alignment_dict = pt.read_json(json_path)
            words = self.get_words(alignment_dict)     
        
            self.bh.segment_by_breath(alignment_id, conversation_id, words, save_folders, max_segment_length, min_segment_length, print_segments=False, save_segments=True)
     
        
    #********************************************* ANALYSIS
    def print_audio_segment(self, start, end, conversation_id):
        newAudio = pt.get_audio_segment(start, end, self.audios[conversation_id])        
        pt.print_audio_segment(start, end, conversation_id, self.audios[conversation_id], self.TEMP_WAVS_DIR)            
        return newAudio
    
    def search_inconsistencies(self, conversation_id, alignment_dict, nonstop = True, incons_threshold = 0.00001): 
        time_factor = pt.get_time_factor(self.aligner)
        
        count_inconsistence = 0
        count_wrong_order = 0
        
        words = self.get_words(alignment_dict)
        for i in range(len(words)):
            inconstistence = False
            wrong_order = False
            
            if not nonstop:
                ipd.clear_output()

            word1, word2 = pt.access_consecutive_words(i, words)
            if word2 == '<end_token>':
                break

            start1, end1, text1 = pt.get_info(word1)
            start2, end2, text2 = pt.get_info(word2)

            start1 = pt.time2seconds(start1, time_factor)
            start2 = pt.time2seconds(start2, time_factor)
            end1 = pt.time2seconds(end1, time_factor)
            end2 = pt.time2seconds(end2, time_factor)
            
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
            self.print_audio_segment(start1, start1+10, conversation_id)
            
            if not nonstop:
                stop = input("Continue? [(y)/n] ")
                if stop == "n":
                    break
        
        print("************************************************")
        print("Total incostistencies:", count_inconsistence)
        print("Total wrong orders:", count_wrong_order)
        
    def breath_stats(self, alignments_relative_path, singleprint = True): 
        # search for breaths across an alignment directory and gives statistics about them
        # alignment_dir = the general directory of the alignment, not the json one.
        time_factor = pt.get_time_factor(self.aligner)
        
        jsons_path = self.ALIGN_DIR + alignments_relative_path
        json_names = [f for f in os.listdir(jsons_path) if os.path.isfile(os.path.join(jsons_path, f))]
        
        breath_stats = {'counts':[], 'averages':[], 'averages_nobreath':[], 'maxs':[], 'mins':[], 'maxs_nobreath':[], 'mins_nobreath':[], 'filestats':{}}
        for i, filename in enumerate(json_names): 
            print(i+1, '/', len(json_names))
            print("Searching in:", filename)
            conversation_id = filename[:-5]
            
            alignment_dict = pt.read_json(jsons_path + filename)
            words = self.get_words(alignment_dict)
            first_word_start, _, _ = pt.get_info(self.aligner, words[0])
            
            count_potential_breath = 0
            count_breath_sections = 0
            
            breath_durations = list()
            nobreath_durations = list()
            start_nobreath = first_word_start
            for i in range(len(words)):
        
                word1, word2 = pt.access_consecutive_words(i, words)
                
                start1, end1, text1 = pt.get_info(self.aligner, word1)
                
                if word2 == '<end_token>':
                    break
                start2, end2, text2 = pt.get_info(self.aligner, word2)
                
                # if there is no breathing in the space:
                if not bh.breathing_in_space(self.aligner, word1, word2, conversation_id):
                    continue
                
                count_potential_breath += 1
        
                start1 = pt.time2seconds(start1, time_factor)
                start2 = pt.time2seconds(start2, time_factor)
                end1 = pt.time2seconds(end1, time_factor)
                end2 = pt.time2seconds(end2, time_factor)        
                
                # Find breath inside the space
                space_audio = pt.get_audio_segment(end1, start2, self.audios[conversation_id])
                breath_section = bh.get_breath_sections(space_audio)
                
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
        max_breaths_i, max_breaths = pt.argmax_max(breath_stats['counts'])
        min_breaths_i, min_breaths = pt.argmin_min(breath_stats['counts'])
        max_duration_i, max_duration_total = pt.argmax_max(breath_stats['maxs'])
        min_duration_i, min_duration_total = pt.argmin_min(breath_stats['mins'])
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
        
    
        