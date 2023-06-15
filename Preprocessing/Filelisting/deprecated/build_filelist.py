# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 14:30:57 2023

@author: nicol
"""
import sys
import argparse
import os
from os.path import isfile, join

import __libpath
from psychelibrary import nic_dataset_tools as ndt

interspeech_dataset_dir_relative = "INTERSPEECH/ComParE2020_Breathing/"
interspeech_breath_labels_dir_relative = 'lab/labels.csv'
interspeech_transcriptions_dir_relative = 'transcriptions_and_alignments/'
interspeech_wavs_dir_relative = "normalized_wav/"
interspeech_temp_wavs_dir_relative = "temp_wav/"

DATASET_DIR, TRANSCR_DIR, WAVS_DIR, TEMP_WAVS_DIR, BREATH_DF, WAV_FILENAMES = ndt.setup_global_paths(
    interspeech_dataset_dir_relative,
    interspeech_transcriptions_dir_relative,
    interspeech_wavs_dir_relative,
    interspeech_temp_wavs_dir_relative,
    interspeech_breath_labels_dir_relative) # Here it uses the default ones

def interspeech_speaker_id(transcript_filename):
    for i, filename in enumerate(WAV_FILENAMES):
        filename = filename[:-4]
        if filename in transcript_filename:
            return i
    
def main(args):
    preposition = "data/" + args.data_name + "/"
    
    save_path = TRANSCR_DIR + args.transcr_dir + 'filelist/'
    exists = os.path.exists(save_path)
    if not exists:
       # Create the save directory
       os.makedirs(save_path)
    with open(save_path + args.data_name + '_filelist.txt', 'w') as f:
        f.write("") # reset the filelist
       
    full_path = TRANSCR_DIR + args.transcr_dir
    filenames = [f for f in os.listdir(full_path) if isfile(join(full_path, f))]
    
    for i, filename in enumerate(filenames): 
        print(i+1, '/', len(filenames), "-", filename)
        file_preposition = preposition + filename[:-4] + ".wav|"
        
        if not args.fake_single_speaker:
            if args.dataset_name == 'INTERSPEECH':
                speaker_id = interspeech_speaker_id(filename)
        else:
            speaker_id = 0
        
        with open(full_path + filename, 'r') as f:
            transcript = f.read()
        
        with open(save_path + args.data_name + '_filelist.txt', 'a') as f:
            f.write(file_preposition + transcript + '|' + str(speaker_id)) # save to the filelist file
            if i < len(filenames)-1:
                f.write("\n")
    
    print("Saved in:", save_path)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--transcr_dir', type = str, default = 'gentle_mfa_assemblyAI/breath1/token_augmented/segmented_alignment/',
        help = 'The directory of the transcriptions to consider.')
    
    parser.add_argument('--data_name', type = str, default = 'INTERSPEECH_segmented',
        help = 'The name to give to the specific filelist dataset.')
    
    parser.add_argument('--fake_single_speaker', type = bool, default = False,
        help = 'If the filelists should specify the speaker for each transcription or fake being them all from a single speaker.')
    
    parser.add_argument('--dataset_name', type = str, default = 'INTERSPEECH',
        help = 'Name of the general dataset.')

    args = parser.parse_args()
    main(args)