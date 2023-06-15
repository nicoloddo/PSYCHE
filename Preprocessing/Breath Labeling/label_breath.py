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

import __libpath
from psychelibrary import nic_dataset_tools as ndt

_, _, ALIGN_DIR, _, _, _, _ = ndt.setup_global_paths() # Here it uses the default ones

BREATH_TOKEN = ndt.BREATH_TOKEN

setting_set = 'default'
with open('../../psychelibrary/Settings and Secrets/breath_labeling_settings.json', 'r') as f:
    breath_settings = json.load(f)[setting_set]

respiration_length_min = float(breath_settings['respiration_length_min'])
interval_db_max = int(breath_settings['interval_db_max'])
interval_peak_db_max = int(breath_settings['interval_peak_db_max'])
respiration_db_max = int(breath_settings['respiration_db_max'])

time_factor = 1

def main(args):    
    ndt.ALIGNER = args.aligner
    ndt.DATASET = args.dataset
    
    full_json_path = ALIGN_DIR + args.alignment_dir + 'json/'
    full_save_path = ALIGN_DIR + args.alignment_dir + 'breath/'
    
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
            if not ndt.breathing_in_space(word1, word2, wav_filename, time_factor, respiration_length_min, interval_db_max, interval_peak_db_max): # using the default threshold of ndt
                continue
            
            count_potential_breath += 1
    
            start1 = ndt.time2seconds(start1, time_factor)
            start2 = ndt.time2seconds(start2, time_factor)
            end1 = ndt.time2seconds(end1, time_factor)
            end2 = ndt.time2seconds(end2, time_factor)        
            
            # Find breath inside the space
            space_audio = ndt.get_audio_segment(end1, start2, wav_filename)
            breath_section = silence.detect_silence(space_audio, min_silence_len=int(respiration_length_min*1000), silence_thresh=respiration_db_max, seek_step = 1)
            
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

    parser.add_argument('--alignment_dir', type = str, default = 'iemocap_default/',
        help = 'The directory of the alignments. Files will be saved in a folder named "breath" in this same directory.')
    
    parser.add_argument('--aligner', type = str, default = 'iemocap_default',
        help = 'The aligner used.')
    
    parser.add_argument('--dataset', type = str, default = 'IEMOCAP',
        help = 'The aligner used.')

    parser.add_argument('--time_factor', type = int, default = 1,
        help = 'The factor to multiply the alignment timestamp to obtain seconds. Set this to 0.001 if your alignments are in ms..')

    args = parser.parse_args()
    main(args)