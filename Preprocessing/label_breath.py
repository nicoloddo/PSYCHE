# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 19:30:54 2023

@author: nicol
"""
import argparse
import os
from os.path import isfile, join
import json

from pydub import AudioSegment, silence

import nic_dataset_tools as ndt

interspeech_dataset_dir_relative = "INTERSPEECH/ComParE2020_Breathing/"
interspeech_breath_labels_dir_relative = 'lab/labels.csv'
interspeech_transcriptions_dir_relative = 'transcriptions_and_alignments/'
interspeech_wavs_dir_relative = "normalized_wav/"
interspeech_temp_wavs_dir_relative = "temp_wav/"

DATASET_DIR, TRANSCR_DIR, WAVS_DIR, TEMP_WAVS_DIR, _, WAV_FILENAMES = ndt.setup_global_paths(
    interspeech_dataset_dir_relative,
    interspeech_transcriptions_dir_relative,
    interspeech_wavs_dir_relative,
    interspeech_temp_wavs_dir_relative,
    interspeech_breath_labels_dir_relative) # Here it uses the default ones


BREATH_TOKEN = ndt.BREATH_TOKEN

ndt.ALIGNER = 'gentle_aligner'

def main(args):
    
    full_json_path = TRANSCR_DIR + args.alignment_dir + 'json/'
    full_save_path = TRANSCR_DIR + args.alignment_dir + 'breath/'
    
    json_names = [f for f in os.listdir(full_json_path) if isfile(join(full_json_path, f))]
    
    breath_stats = {'counts':[], 'averages':[], 'maxs':[], 'mins':[]}
    for filename in json_names: 
        print(filename)
        wav_filename = filename[:-5] + '.wav'
        
        with open(full_json_path + filename, 'r') as f:
            data = json.load(f)
            alignment_json = dict(data)
        words = ndt.get_words_json(alignment_json)
        
        count_potential_breath = 0
        count_breath_sections = 0
        
        breath_alignment = list()
        transcript = ''
        breath_durations = list()
        for i in range(len(words)):
    
            word1, word2 = ndt.access_consecutive_words(i, alignment_json)
            
            start1, end1, text1 = ndt.get_info(word1)
            
            # Add first word (the second will be added at next iteration)
            breath_alignment.append(word1)
            transcript += text1 + ' '
            
            if word2 == '<end_token>':
                break
            start2, end2, text2 = ndt.get_info(word2)
            
            # if there is no breathing in the space:
            if not ndt.breathing_in_space(word1, word2, wav_filename, args.time_factor, args.respiration_length_min, args.interval_db_max, args.interval_peak_db_max): # using the default threshold of ndt
                continue
            
            count_potential_breath += 1
    
            start1 = ndt.time2seconds(start1, args.time_factor)
            start2 = ndt.time2seconds(start2, args.time_factor)
            end1 = ndt.time2seconds(end1, args.time_factor)
            end2 = ndt.time2seconds(end2, args.time_factor)        
            
            # Find breath inside the space
            space_audio = ndt.get_audio_segment(end1, start2, wav_filename)
            breath_section = silence.detect_silence(space_audio, min_silence_len=int(args.respiration_length_min*1000), silence_thresh=args.respiration_db_max, seek_step = 1)
            
            count_breath_sections += len(breath_section)
            for breath in breath_section:
                start_timestamp = end1 + breath[0]/1000
                end_timestamp = end1 + breath[1]/1000
                duration = end_timestamp - start_timestamp # in seconds
                breath_durations.append(round(duration, 2))
                
                breath_insert = {'start': start_timestamp, 'end': end_timestamp, 'word':BREATH_TOKEN, 'case':'breath'}
                
                # Add breaths
                breath_alignment.append(breath_insert)
                transcript += BREATH_TOKEN + ' '
        
        average = round(sum(breath_durations)/len(breath_durations), 2)
        max_duration = max(breath_durations)
        min_duration = min(breath_durations)
        breath_stats['counts'].append(count_breath_sections)
        breath_stats['averages'].append(average)
        breath_stats['maxs'].append(max_duration)
        breath_stats['mins'].append(min_duration)
        
        print("Potential breath sections:", count_potential_breath)
        print("Actual breath sections:", count_breath_sections)
        print("Average breath duration:", average)
        print("Max breath duration:", max_duration)
        print("Min breath duration:", min_duration)
        
        # save alignments
        breath_alignment_dict = {'words':breath_alignment}
        with open(full_save_path + 'json/' + filename, 'w') as f:
            json.dump(breath_alignment_dict, f, indent = 1)
        
        # save transcription
        transcript = transcript[:-1] # Because we are adding a space at the end with the for loop
        with open(full_save_path + filename[:-5] + '.txt', 'w') as f:
            f.write(transcript)
                
        print("************************************************")
        print()
    
    count_breath_average = round(sum(breath_stats['counts']) / len(breath_stats['counts']), 2)
    max_breaths = max(breath_stats['counts'])
    min_breaths = min(breath_stats['counts'])
    total_duration_average = sum(breath_stats['averages']) / len(breath_stats['averages'])
    print("FINAL STATS:")
    print("Average number of breaths:", count_breath_average)
    print("Max breaths:", max_breaths)
    print("Min breaths:", min_breaths)
    print("Average breath duration:", total_duration_average)
    print("Max breath duration:", max(breath_stats['maxs']))
    print("Min breath duration:", min(breath_stats['mins']))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--alignment_dir', type = str, default = 'gentle_mfa_assemblyAI/',
        help = 'The directory of the alignments. Files will be saved in a folder named "breath" in this same directory.')
    
    parser.add_argument('--respiration_length_min', type = float, default = 0.19,
        help = 'The minimum length of a breath instance to be detected.')
    
    parser.add_argument('--interval_db_max', type = int, default = -0,
        help = 'The maximum db of a pause in the speaking to be a potential breath pause.')
    
    parser.add_argument('--interval_peak_db_max', type = int, default = -0,
        help = 'The maximum of the peak of dbs in a pause to be a potential breath pause.')
    
    parser.add_argument('--respiration_db_max', type = int, default = -40,
        help = 'The maximum db of a breath instance.')

    parser.add_argument('--time_factor', type = int, default = 1,
        help = 'The factor to multiply the alignment timestamp to obtain seconds. Set this to 0.001 if your alignments are in ms..')

    args = parser.parse_args()
    main(args)