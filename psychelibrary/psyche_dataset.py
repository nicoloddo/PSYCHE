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
        
    def parse_alignment_file(self, filename, alignment_dir):
        
        full_json_path = self.ALIGN_DIR + alignment_dir + 'json/'
        
        with open(full_json_path + filename, 'r') as f:
            data = json.load(f)
            alignment_json = dict(data)
        words = ndt.get_words_json(alignment_json)
        
        self.alignments[alignment_dir] = words
    
    def parse_alignment_turn_iemocap(self, row):
        '''
        This function can be used if the dataset is IEMOCAP, because in the IEMOCAP we have one turn per row instead of one conversation per row

        Returns
        -------
        None.

        '''
        if self.dataset_name != 'IEMOCAP':
            sys.exit("Dataset has to be IEMOCAP for this function.")
            
        turn_words = ndt.parse_iemocap_turn_alignment(row['alignment_path'])
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
        
        for turn_word in flat_dialog_words:
            start_time_of_turn = float(dialog_subset.loc[dialog_subset['turn_name'] == turn_word['turn'], 'start_time'].values[0])
            actor =  dialog_subset.loc[dialog_subset['turn_name'] == turn_word['turn'], 'actor_speaking_id'].values[0]
            turn_word['start'] = float(turn_word['start']) + start_time_of_turn
            turn_word['end'] = float(turn_word['end']) + start_time_of_turn
            turn_word['actor'] = actor
            
        return sorted(flat_dialog_words, key=lambda d: d['start'])
        
    
    def parse_alignment_audio(self, row):
        '''
        Parses alignment of an audio: one audio per row. In case of IEMOCAP it parses the one of one full dialog.
        Remember that it will repeat this for every turn if you pass the full iemocap dataset.

        Returns
        -------
        None.

        '''
        if self.dataset_name == 'IEMOCAP':
            dialog_subset = self.df.loc[self.df['conversation_id'] == row['conversation_id']]
            return self.parse_alignment_iemocap_dialog(dialog_subset)
        else:
            pass
        
        
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
        
        return np.array(reduced_noise1, dtype=np.int16), rate1, np.array(reduced_noise2, dtype=np.int16), rate2
        
    def separate_speaker_channels(self):
        self.df.apply(self.separate_channels_row, axis=1)
        
    def diarize(self, aligner):
        pass
        