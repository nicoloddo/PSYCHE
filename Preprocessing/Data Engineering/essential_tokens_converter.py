# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:35:56 2023

@author: nicol
"""
import sys
import argparse
import os
from os.path import isfile, join

import json

from common import dataset_tools as ndt

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

ndt.ALIGNER = 'gentle_aligner'

DISFLUENCIES = ['uh', 'ah', 'eh', 'oh', 'um', 'em']
def main(args):
    save_path = TRANSCR_DIR + args.breath_alignment_dir + 'token_augmented/'
    exists = os.path.exists(save_path)
    if not exists:
       # Create the save directory
       os.makedirs(save_path)
       os.makedirs(save_path + 'json/')
       
    full_json_path = TRANSCR_DIR + args.breath_alignment_dir + 'json/'
    json_names = [f for f in os.listdir(full_json_path) if isfile(join(full_json_path, f))]
    
    for i, filename in enumerate(json_names): 
        print(i+1, '/', len(json_names))
        print("Converting:", filename)
        
        with open(full_json_path + filename, 'r') as f:
            data = json.load(f)
            alignment_json = dict(data)
        words = ndt.get_words_json(alignment_json)
        
        token_transcript = ""
        for j, word in enumerate(words):
            _, _, text = ndt.get_info(word)
            if text.lower() in DISFLUENCIES:
                word['token'] = '{disfl.' + text.lower() + '}'
                word['converted'] = word['token']
            elif not text == args.breath_token:
                word['token'] = '{word}'
                if args.convert_words:
                    word['converted'] = word['token']
                else:
                    word['converted'] = text.lower()
            elif text == args.breath_token:
                word['token'] = '{breath}'
                word['converted'] = word['token']
            else:
                sys.exit("Error: a word is neither a normal word, a disfluency or a breath.")
            
            token_transcript += word['converted'] + ' '
        
        token_transcript = token_transcript[:-1] # to delete the last space
        if args.convert_words:
            converted_filename = filename[:-5] + '_essential'
        else:
            converted_filename = filename[:-5]
        with open(save_path + 'json/' + converted_filename + '.json', 'w') as f:
            json.dump({'words':words}, f, indent = 1)
        with open(save_path + converted_filename + '.txt', 'w') as f:
            f.write(token_transcript) # save the transcription
    
    print("Saved in:", save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--breath_alignment_dir', type = str, default = 'gentle_mfa_assemblyAI/breath1/',
        help = 'The directory of the alignments. The augmented alignment will be saved in a folder named "token_augmented" in the directory of the breath set.')
    
    parser.add_argument('--convert_words', type = bool, default = False,
        help = 'If the script is supposed to convert all words to a token named {word}.')
    
    parser.add_argument('--breath_token', type = str, default = ndt.BREATH_TOKEN,
        help = 'The breath token used in the transcription.')

    args = parser.parse_args()
    main(args)