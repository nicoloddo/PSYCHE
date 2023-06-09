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

import sys
import os
import warnings
import json

from psychelibrary import nic_dataset_tools as ndt

class PsycheDataset():
    '''
    Wrapper around Datasets to be used in the PSYCHE project
    '''
    
    def __init__(self, dataset_name, dataframe):
        ndt.DATASET = dataset_name
        _, _, self.ALIGN_DIR, _, _, _, _ = ndt.setup_global_paths() # Here it uses the default ones
        
        self.dataset_name = dataset_name
        self.df = dataframe
        
        self.aligner = None
        self.alignments = {}
        
        
    def set_aligner(self, aligner):
        ndt.ALIGNER = aligner
        self.aligner = aligner
        
    def parse_alignment_file(self, filename, alignment_id):
        
        full_json_path_file = self.ALIGN_DIR + alignment_id + '/' + 'json/' + filename
        
        if not os.path.isfile(full_json_path_file):
            warnings.warn("\n[In parse_alignment_file()]: Some alignment files are not found.")
            return None
            
        with open(full_json_path_file, 'r') as f:
            data = json.load(f)
            alignment_json = dict(data)
        words_alignment = ndt.get_words_json(alignment_json)
        
        return words_alignment
    
    def parse_alignment_turn_iemocap(self, row):
        '''
        This function can be used if the dataset is IEMOCAP, because in the IEMOCAP we have one turn per row instead of one conversation per row

        Returns
        -------
        None.

        '''
        if self.dataset_name != 'IEMOCAP':
            sys.exit("Dataset has to be IEMOCAP for this function.")
        
        alignment_path = row['alignment_path']
        turn_words = ndt.parse_iemocap_turn_alignment(alignment_path)
        return turn_words
    
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
        
    def separate_channels_row(self, row): #!!! Still testing
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
        audio = AudioSegment.from_wav(row['wav_path'])

        # Cut the audio to the required duration
        start_time_ms = row['start_time'] * 1000  # start_time should be in seconds
        end_time_ms = row['end_time'] * 1000  # end_time should be in seconds
        audio = audio[start_time_ms:end_time_ms]

        # Split stereo to mono
        channels = audio.split_to_mono()

        # Denoise each channel
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
        
        return np.array(reduced_noise1, dtype=np.int16), rate1, np.array(reduced_noise2, dtype=np.int16), rate2
        
    def separate_speaker_channels(self):
        self.df.apply(self.separate_channels_row, axis=1)
        
    def diarize(self, aligner):
        pass
        