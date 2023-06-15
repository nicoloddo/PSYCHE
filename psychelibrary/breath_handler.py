# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:50:31 2023

@author: nicol
"""

import sys

import json

from pydub import AudioSegment, silence
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np # linear algebra

from psychelibrary import psyche_tools as pt

class BreathHandler():
    
    def __init__(self, breathing_settings_set, settings_directory, audios, temp_wavs_dir, breath_df = None):
        with open(settings_directory + 'breath_labeling_settings.json', 'r') as f:
            breath_settings = json.load(f)[breathing_settings_set]
        
        self.breath_df = breath_df
        self.audios = audios
        self.breathing_settings_set = breathing_settings_set
        self.respiration_length_min = float(breath_settings['respiration_length_min']) # in seconds
        self.interval_db_max = int(breath_settings['interval_db_max']) # in db
        self.interval_peak_db_max = int(breath_settings['interval_peak_db_max']) # in db
        self.respiration_db_max = int(breath_settings['respiration_db_max']) # in db
        self.breath_token = breath_settings['breath_token']
        self.temp_wavs_dir = temp_wavs_dir
                
    def breathing_in_space(self, aligner, word1, word2, conversation_id):
        
        time_factor = pt.get_time_factor(aligner)
        
        start1, end1, _ = pt.get_info(aligner, word1)
        start2, end2, _ = pt.get_info(aligner, word2)
        
        start1 = pt.time2seconds(start1, time_factor)
        start2 = pt.time2seconds(start2, time_factor)
        end1 = pt.time2seconds(end1, time_factor)
        end2 = pt.time2seconds(end2, time_factor)
        
        # get audio between words ---------------------------
        start_ms = end1 * 1000 #Works in milliseconds
        end_ms = start2 * 1000
        breathAudio = AudioSegment.from_wav(self.audios[conversation_id])
        breathAudio = breathAudio[start_ms:end_ms]
        
        # check if we should return True: is it (not) breathing? --
        if start2-end1 < self.respiration_length_min:
            return False
        elif breathAudio.dBFS > self.interval_db_max:
            return False
        elif breathAudio.max_dBFS > self.interval_peak_db_max:
            return False
        else:
            return True
        
    def breathing_in_word(self, aligner, word, conversation_id):
        
        time_factor = pt.get_time_factor(aligner)
        
        start, end, _ = pt.get_info(aligner, word)
        start = pt.time2seconds(start, time_factor)
        end = pt.time2seconds(end, time_factor)
        
        # get audio between words ---------------------------
        start_ms = start * 1000 #Works in milliseconds
        end_ms = end * 1000
        breathAudio = AudioSegment.from_wav(self.audios[conversation_id])
        breathAudio = breathAudio[start_ms:end_ms]
        
        # check if we should return True: is it (not) breathing? --
        if end-start < self.respiration_length_min:
            return False
        elif breathAudio.dBFS > self.interval_db_max:
            return False
        elif breathAudio.max_dBFS > self.interval_peak_db_max:
            return False
        else:
            return True
        
    def get_breath_sections(self, space_audio):
        return silence.detect_silence(space_audio, min_silence_len=int(self.respiration_length_min*1000), silence_thresh=self.respiration_db_max, seek_step = 1)
        
    def print_audio_segment(self, start, end, conversation_id, temp_wavs_dir):
        newAudio = pt.get_audio_segment(start, end, self.audios[conversation_id])        
        pt.print_audio_segment(start, end, conversation_id, self.audios[conversation_id], temp_wavs_dir)            
        return newAudio
        
    def print_breath_df_word(self, i, conversation_id, json):
        words = self.get_words_json(json)
        # json informations on the word: --------------------
        print(words[i])

        # Breathing signal:----------------------------------
        start, end, text = self.get_info(words[i])
        
        start = pt.time2seconds(start, self.time_factor)
        end = pt.time2seconds(end, self.time_factor)
        
        if not self.breath_df.empty:
            df = self.breath_df.loc[self.df['filename'] == conversation_id + '.wav']
        
            df = df.loc[df['timeFrame'] >= start]
            df = df.loc[df['timeFrame'] <= end]
        
            print()
            print("Word:", text)
            print()
            print(df)
            plt.figure(figsize=(10,5))
            plt.plot(df['timeFrame'], df['upper_belt'])
            plt.xticks(np.arange(start, end, 0.1), rotation = 45)
            plt.yticks(np.arange(min(self.breath_df['upper_belt']), max(self.breath_df['upper_belt']), 0.1), rotation = 45)
            plt.grid(True)
            plt.show()

        # Audio:--------------------------------------------
        start_ms = start * 1000 #Works in milliseconds
        end_ms = end * 1000
        newAudio = AudioSegment.from_wav(df.iloc[0]['conversation_wav_path'])
        newAudio = newAudio[start_ms:end_ms]

        newAudio.export(self.temp_wavs_dir + 'temp.wav', format="wav")
        ipd.display(ipd.Audio(self.temp_wavs_dir + 'temp.wav'))
        
    def print_breath_df_spaced_words(self, aligner, word1, word2, conversation_id, images = True, collective_image = True, only_space_audio = False, original_audio = False, printdf = False, printjson = True):
        time_factor = pt.get_time_factor(aligner)
        filename = conversation_id + '.wav'
        
        start1, end1, text1 = pt.get_info(word1)
        start2, end2, text2 = pt.get_info(word2)
        
        # convert into seconds (if necessary)
        start1 = pt.time2seconds(start1, time_factor)
        start2 = pt.time2seconds(start2, time_factor)
        end1 = pt.time2seconds(end1, time_factor)
        end2 = pt.time2seconds(end2, time_factor)
        
        text = text1 + ' _ ' + text2
        
        # json informations on the word: --------------------
        if printjson:
            print(word1)
            print()
            print(word2)
            print()

        # Breathing signal:----------------------------------
        if not self.breath_df.empty:
            df_file = self.breath_df.loc[self.breath_df['filename'] == filename]
        
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
        
        if not self.breath_df.empty:
            if images:
                plt.figure(figsize=(10,5))
                plt.plot(df['timeFrame'], df['upper_belt'])
                plt.xticks(np.arange(start1, end2, 0.1), rotation = 45)
                plt.yticks(np.arange(min(self.breath_df['upper_belt']), max(self.breath_df['upper_belt']), 0.1), rotation = 45)
                plt.grid(True)
                plt.axvline(x = end1, color = 'b', label = 'axvline - full height')
                plt.axvline(x = start2, color = 'b', label = 'axvline - full height')
                plt.show()
            
            if not collective_image:
                plt.figure(figsize=(120,5))
                plt.plot(df_file['timeFrame'], df_file['upper_belt'])
                plt.xticks(np.arange(min(df_file['timeFrame']), max(df_file['timeFrame'])+1, 0.5), rotation = 45)
                plt.yticks(np.arange(min(self.breath_df['upper_belt']), max(self.breath_df['upper_belt']), 0.1), rotation = 45)
                plt.grid(True)
                plt.axvline(x = start1, color = 'r', label = 'axvline - full height')
                plt.axvline(x = end1, color = 'b', label = 'axvline - full height')
                plt.axvline(x = start2, color = 'b', label = 'axvline - full height')
                plt.axvline(x = end2, color = 'r', label = 'axvline - full height')
                plt.show()

        if original_audio:
            ipd.display(ipd.Audio(self.audios[conversation_id]))

        # Audio:--------------------------------------------
        print("Section:")
        self.print_audio_segment(start1, end2, conversation_id)

        if not only_space_audio:
            # Audio:--------------------------------------------
            print("First word:")
            self.print_audio_segment(start1, end1, conversation_id)

            # Audio:--------------------------------------------
            print("Second word:")
            self.print_audio_segment(start2, end2, conversation_id)
        
        # Space Audio:--------------------------------------------
        print("Space in between:")
        self.print_audio_segment(end1, start2, conversation_id)
        
    def search_breath_analysis(self, aligner, conversation_id, words, nonstop=False, printonlyfinal=False, printjson = False, askcontinue=False):
        time_factor = pt.get_time_factor(aligner)
        filename = conversation_id + '.wav'
        
        if nonstop:
            if not self.breath_df.empty:
                df_file = self.breath_df.loc[self.breath_df['filename'] == filename]
                plt.figure(figsize=(120,5))
                plt.plot(df_file['timeFrame'], df_file['upper_belt'])
                plt.xticks(np.arange(min(df_file['timeFrame']), max(df_file['timeFrame'])+1, 0.5), rotation = 45)
                plt.yticks(np.arange(min(self.breath_df['upper_belt']), max(self.breath_df['upper_belt']), 0.1), rotation = 45)
                plt.grid(True)
        else:
            printonlyfinal = False
        
        count_potential_breath = 0
        count_breath_sections = 0
        
        for i in range(len(words)):
            if not nonstop:
                ipd.clear_output()

            word1, word2 = pt.access_consecutive_words(i, words)
            if word2 == '<end_token>':
                break

            if not self.breathing_in_space(aligner, word1, word2, conversation_id):
                continue
            
            start1, end1, text1 = pt.get_info(word1)
            start2, end2, text2 = pt.get_info(word2)

            start1 = pt.time2seconds(start1, time_factor)
            start2 = pt.time2seconds(start2, time_factor)
            end1 = pt.time2seconds(end1, time_factor)
            end2 = pt.time2seconds(end2, time_factor)
            
            
            count_potential_breath += 1
            
            if not printonlyfinal:
                print("i:", i)
                print("Potential breath spaces:", count_potential_breath)
                self.print_breath_df_spaced_words(aligner, word1, word2, conversation_id, images = not nonstop, collective_image = nonstop, only_space_audio= nonstop, printjson = printjson)
                print()
                
            # Find breath inside the space
            space_audio = pt.get_audio_segment(end1, start2, self.audios[conversation_id])
            breath_section = silence.detect_silence(space_audio, min_silence_len=int(self.respiration_length_min*1000), silence_thresh=self.respiration_db_max, seek_step = 1)
            
            
            if len(breath_section) > 0: # if breath has been detected
                count_breath_sections += len(breath_section)
                if nonstop: # place red bar for the audio section
                    if not self.breath_df.empty:
                        plt.axvline(x = start1, color = 'r', label = 'axvline - full height')
                        plt.axvline(x = end2, color = 'r', label = 'axvline - full height')
                
                if len(breath_section) == 1 and round((breath_section[0][1] - breath_section[0][0])/1000, 3) == round(start2-end1, 3):
                    if not printonlyfinal:
                        print("THE WHOLE SPACE IS A BREATH.")                
                    if nonstop: # place blue bars for the breath sections
                        if not self.breath_df.empty:
                            plt.axvline(x = end1, color = 'b', label = 'axvline - full height')
                            plt.axvline(x = start2, color = 'b', label = 'axvline - full height')
                        
                else:
                    if not printonlyfinal:
                        print("BREATH SECTIONS:")
                    for breath in breath_section:
                        start_timestamp = end1 + breath[0]/1000
                        end_timestamp = end1 + breath[1]/1000
                        
                        if not printonlyfinal:
                            self.print_audio_segment(start_timestamp, end_timestamp, conversation_id)
                            print(breath)
                            print("Duration:", end_timestamp - start_timestamp, "seconds")
                            print()
                        
                        if nonstop: # place blue bars for the breath sections
                            if not self.breath_df.empty:
                                plt.axvline(x = start_timestamp, color = 'b', label = 'axvline - full height')
                                plt.axvline(x = end_timestamp, color = 'b', label = 'axvline - full height')
            
            else: # there are no breaths
                if not printonlyfinal:
                    print("NO BREATH FOUND IN THIS SPACE SECTION.")
                    if nonstop: # place red bar for the audio section even though there is no breath
                        if not self.breath_df.empty:
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
            if not self.breath_df.empty:
                plt.show()
    
    def segment_files_by_breath(self, aligner, conversation_id, words, save_folders, max_segment_length, min_segment_length, print_segments = True, save_segments = False): 
        
        i = 0
        start, end, text = pt.get_info(words[i], aligner) # the initialization is the start of the first word or breath in the alignment
        segment_start_index = i # the index of the first word of segment
        segment_start = start
        
        for i in range(1, len(words)):
            start, end, text = pt.get_info(words[i]) # select next word
            
            if end - segment_start > max_segment_length: # if the segment has reached the max length without finding a breath
                self.segment_by_index(aligner, conversation_id, words, save_folders, segment_start_index, i-1, print_segments, save_segments) # save stopping at the previous word
                segment_start_index = i # set the current word (breath) as start index of next segment
                segment_start = start
                
            elif text == self.breath_token: # if it is a breath token
                if end - segment_start < min_segment_length: # if the segment is too little, we go search the next breath
                    continue
                else: # the segment is not too little and there is a breath token
                    self.segment_by_index(aligner, conversation_id, words, save_folders, segment_start_index, i, print_segments, save_segments) # save
                    segment_start_index = i # set the current word (breath) as start index of next segment
                    segment_start = start            
                
        
    def segment_by_index(self, aligner, conversation_id, words, save_folders, start_index, end_index, print_segments = True, save_segments = False, token_segment = True):
        save_filenames = [
            self.breathing_settings_set + 'segmented_alignment/json/' + conversation_id + '_' + str(start_index) + '-' + str(end_index) + '.txt',
            self.breathing_settings_set + 'segmented_alignment/' + conversation_id + '_' + str(start_index) + '-' + str(end_index) + '.txt',
            self.breathing_settings_set + 'segmented_tokens/' + conversation_id + '_' + str(start_index) + '-' + str(end_index) + '.txt',
            self.breathing_settings_set + 'segmented_wav/' + conversation_id + '_' + str(start_index) + '-' + str(end_index) + '.wav',
            ]
        
        save_transcr_dir = save_folders[0] + save_filenames[0]
        save_json_dir = save_folders[1] + save_filenames[1]
        save_token_transcr_dir = save_folders[2] + save_filenames[2]
        save_wav_dir = save_folders[3] + save_filenames[3]
        
        segm_transcr = ''
        segm_token_transcr = ''
        segm_alignment = {'words':list()}
        start_segm, end_segm, text = pt.get_info(aligner, words[start_index]) # first word
        for i in range(start_index, end_index + 1): # without the +1 end_index would be skipped
            segm_alignment['words'].append(words[i])
            start, end, text = pt.get_info(words[i])
            if 'converted' not in words[i]:
                sys.exit("Error: you did not run the token converter.")
            segm_transcr += words[i]['converted'] + ' '
            if token_segment:
                segm_token_transcr += words[i]['token'] + ' '
            
        if print_segments:
            print(segm_transcr)
            print("Duration:", round(end - start_segm, 2))
            print()
            self.print_audio_segment(start_segm, end, conversation_id)
        
        if save_segments:
            with open(save_json_dir, 'w') as f:
                json.dump(segm_alignment, f, indent = 1) # save the alignment
            with open(save_transcr_dir, 'w') as f:
                f.write(segm_transcr[:-1]) # save the transcription
            if token_segment:
                with open(save_token_transcr_dir, 'w') as f:
                    f.write(segm_token_transcr[:-1]) # save the token transcription
            segmAudio = pt.get_audio_segment(start_segm, end, self.audios[conversation_id])
            segmAudio.export(save_wav_dir, format="wav") # save the audio
