# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:07:21 2023

@author: nicol
"""
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


BREATH_TOKEN = ndt.BREATH_TOKEN

ndt.ALIGNER = 'gentle_aligner'


def main(args):
    full_json_path = TRANSCR_DIR + args.breath_alignment_dir + 'json/'
    json_names = [f for f in os.listdir(full_json_path) if isfile(join(full_json_path, f))]
    
    for filename in json_names:
        print("Segmenting", filename)
        ndt.segment_files(args.breath_alignment_dir, filename, args.max_segment_length, args.min_segment_length, print_segments=False, save_segments=True)        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--breath_alignment_dir', type = str, default = 'gentle_mfa_assemblyAI/breath1/token_augmented/',
        help = 'The directory of the breath alignments. The segments will be saved in a folder named "wav_segments" in this same directory.')
    
    parser.add_argument('--max_segment_length', type = float, default = 15,
        help = 'The maximum length of each segment (in seconds).')
    
    parser.add_argument('--min_segment_length', type = int, default = 4,
        help = 'The minimum length of each segment (in seconds).')

    args = parser.parse_args()
    main(args)