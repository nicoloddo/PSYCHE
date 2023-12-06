# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:10:37 2023

@author: nicol
"""

import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

import requests
import json

import nic_dataset_tools as ndt

dataset_dir_relative = "INTERSPEECH/ComParE2020_Breathing/"
breath_labels_dir_relative = 'lab/labels.csv'
transcriptions_dir_relative = 'transcriptions/'
alignments_dir_relative = 'alignments/'
wavs_dir_relative = "normalized_wav/"
temp_wavs_dir_relative = "temp_wav/"

DATASET_DIR, TRANSCR_DIR, ALIGN_DIR, WAVS_DIR, TEMP_WAVS_DIR, BREATH_DF, WAV_FILENAMES = ndt.setup_global_paths(
    dataset_dir_relative,
    breath_labels_dir_relative,
    transcriptions_dir_relative,
    alignments_dir_relative,
    wavs_dir_relative,
    temp_wavs_dir_relative) # Here it uses the default ones

def separate_breath_df():
    for filename in WAV_FILENAMES:
        temp_df = BREATH_DF.loc[BREATH_DF['filename'] == filename]
        temp_df.to_csv(DATASET_DIR + 'lab/Separated/' + filename[:-4] + '_label.csv', mode='w', columns=['timeFrame','upper_belt'], index=False)
        
    
def ibm_transcript(url = ndt.IBM_URL, apikey = ndt.IBM_APIKEY):
    for filename in WAV_FILENAMES:
        '''
        if filename == 'devel_00.wav' or filename == 'devel_01.wav': # I did those manually, I don't want to lose minutes of the API
            continue
        '''
    
        headers = {
            'Content-Type': 'audio/wav',
        }
    
        #model = 'en-US_BroadbandModel'
        model = 'en-US_Multimedia' # way better model here! It does not have the esitation marks though.
        params = {
            'timestamps': 'true',
            'max_alternatives': '1',
            'model':model,
        }
    
        with open(WAVS_DIR + filename, 'rb') as f:
            data = f.read()
    
        x = requests.post(url, params=params, headers=headers, data=data, auth=('apikey', apikey))
    
        data = x.json()
        with open(TRANSCR_DIR + 'transcriptions_ibm/json/' + filename[:-4] + '.json', 'w') as f:
            json.dump(data, f, indent = 1)
    
        resp = dict(data)
    
        transcript = ""
        for i in range(len(resp['results'])):
             transcript += resp['results'][i]['alternatives'][0]['transcript']
    
        with open(TRANSCR_DIR + 'transcriptions_ibm_' + model + '/' + filename[:-4] + '.txt', 'w') as f:
            f.write(transcript[:-1]) # To delete the space at the end we do [:-1]
            
        
def assai_queue_transcripts(apikey, count_limit = math.inf, except_filenames = []):
    # Assembly AI requires to put in the queue the audios, and only after to get the transcriptions through another request.
    # This means that before you can even check if everything was fine, all the minutes of transcriptions are already invoiced.
    # Use the count_limit parameter to do some trials before using it.
    
    # count_limit = math.inf would put all the transcriptions in the queue
    
    transcript_ids = {}
    count = 0
    for filename in WAV_FILENAMES:
        if filename in except_filenames: # If you did some manually and don't want to lose minutes of the API
            continue
        
        
        if count >= count_limit:
            break
        print(count + 1)
        
        with open(WAVS_DIR + filename, 'rb') as f:
                data = f.read()
        
        
        headers = {'authorization': apikey}
        
        response = requests.post('https://api.assemblyai.com/v2/upload',
                                headers=headers,
                                data=data)
        
        upload_url = response.json()['upload_url']
        print(upload_url)
        
        endpoint = "https://api.assemblyai.com/v2/transcript"
        json_send = {
            "audio_url": upload_url,
            "disfluencies": True,
            "format_text": True,
            "speaker_labels": True
        }
        headers = {
            "authorization": apikey,
        }
        response = requests.post(endpoint, json=json_send, headers=headers)
        print(response.json()['status'])
        count += 1
        
        transcript_ids[filename] = response.json()['id']
    
    return transcript_ids

def assai_get(transcr_id, apikey):
    endpoint = "https://api.assemblyai.com/v2/transcript/" + transcr_id
    headers = {
        "authorization": apikey,
    }
    response = requests.get(endpoint, headers=headers)
    if response.ok == False:
        print(response.text)
    
    print("Waiting for completion...")
    while response.json()['status'] != 'completed': # Wait for transcript to be completed
        response = requests.get(endpoint, headers=headers)
        
    return response

def assai_save(response, filename):
    # The filename needs the .wav extension
    print("Saving...")
    with open(TRANSCR_DIR + 'transcript_assemblyAI/json/' + filename[:-4] + '.json', 'w') as f:
        json.dump(response.json(), f, indent = 1)
        
    with open(TRANSCR_DIR + 'transcript_assemblyAI/' + filename[:-4] + '.txt', 'w') as f:
            f.write(response.json()['text'])
            
    print("Saved.")
    
def assai_get_transcripts(transcript_ids, apikey):
    for filename in transcript_ids:
        response = assai_get(transcript_ids[filename], apikey)
        assai_save(response, filename)
            
def assemblyai_transcript(count_limit = 1, except_filenames=[]):
    # Use the count_limit parameter to do some trials before using it. Because transcriptions are parallelized here.
    # count_limit = math.inf would put all the transcriptions in the queue
    
    apikey = ndt.ASSAY_APIKEY
    
    transcripts_ids = assai_queue_transcripts(apikey, count_limit, except_filenames) # request transcriptions
    assai_get_transcripts(transcripts_ids, apikey) # get transcriptions and save
    

def gentle_align(transcriptions_dir_name, aligned_dir_name, sorted_dir_name, port = '32768', except_filenames = [], transcr_suffix = ''):
    # Performs the alignment with the gentle aligner. Saves the aligned transcription and the sorted aligned transcription.
    # Gentle does not output the alignment in a sorted by 'start' way.
    # Remember to run the Gentle docker image before and change the port parameter accordingly.
    
    # Example usage:
    # gentle_align('transcript_assemblyAI/', 'aligned_assemblyAI/', 'sorted_aligned_assemblyAI/', '32768')
    
    # Example of except_filenames list: [filename for filename in WAV_FILENAMES if filename.startswith("devel_0")]
    # except_filenames default is ['devel_10.wav', 'test_08.wav', 'train_01.wav', 'train_10.wav', 'train_14.wav'] because those 2 did not work with gentle.
    # This difficult reproducibility is also a reason to avoid gentle and to go for the mfa. MFA also has more options for personalization.
    
    for filename in WAV_FILENAMES:
        if filename in except_filenames: # I did those manually, I don't want to lose minutes of the API
            continue
        
        print(filename, ":")
    
        params = {
            'async': 'false',
            #'disfluency' : 'true',
            #'conservative' : 'true',
        }
    
        files = {
            'audio': open(WAVS_DIR + filename, 'rb'),
            'transcript': open(TRANSCR_DIR + transcriptions_dir_name + filename[:-4] + transcr_suffix + '.txt', 'rb'),
        }
    
        x = requests.post('http://localhost:' + port + '/transcriptions', params=params, files=files)
    
        print("Got alignment.")
        
        # SAVE THE ALIGNED JSON
        data = x.json()
        with open(ALIGN_DIR + aligned_dir_name + 'json/' + filename[:-4] + '.json', 'w') as f:
            json.dump(data, f, indent = 1)
    
        resp = dict(data)
    
        transcript = "" # normal transcription
        valid_align = list() # aligned and sorted by time transcription
        for i in range(len(resp['words'])):
             transcript += resp['words'][i]['word'] + ' ' # add them to the normal transcription
             
             if 'start' in resp['words'][i]: #
                 valid_align.append(resp['words'][i]) # if they dont have start in the keys, it is not a valid word for the sorted transcription
                 
        transcript = transcript[:-1] # Because we are adding a space at the end with the for loop
    
        print("Saved alignment.")
        
        # SAVE THE NORMAL ALIGNED TRANSCRIPTION
        with open(ALIGN_DIR + aligned_dir_name + filename[:-4] + '.txt', 'w') as f:
            f.write(transcript)
    
        # SORTED BY START ALIGNMENT
        # sort the dict by start
        sorted_align = sorted(valid_align, key=lambda d: d['start']) 
    
        transcript = ""
        for i in range(len(sorted_align)):
             transcript += resp['words'][i]['word'] + ' '
        transcript = transcript[:-1] # Because we are adding a space at the end with the for loop
    
        with open(ALIGN_DIR + sorted_dir_name + filename[:-4] + '.txt', 'w') as f:
            f.write(transcript)
        
        sorted_dict = {'words':sorted_align}
        with open(ALIGN_DIR + sorted_dir_name + 'json/' + filename[:-4] + '.json', 'w') as f:
            json.dump(sorted_dict, f, indent = 1)
        
        print("Saved sorted alignment.")
        
def gentle_mfa_align(gentle_align_dir, mfa_align_dir, gentle_mfa_align_dir, except_filenames = ['devel_10.wav', 'test_08.wav', 'train_01.wav', 'train_10.wav', 'train_14.wav']): 
    # Combines Gentle and MFA with a priority on Gentle. It uses Gentle, but when Gentle misses a timestamp, it uses MFA.
    # Moreover it makes sure there are no inconsistencies: the start of each word will always be >= the end of the one before.
    # Example usage:
    # gentle_mfa_align('aligned_assemblyAI/json/', 'MFAFormatCorpus/aligned_assai/', 'gentle_mfa_assemblyAI/')
    # except_filenames default is ['devel_10.wav', 'test_08.wav', 'train_01.wav', 'train_10.wav', 'train_14.wav'] because those did not work with gentle.
    
    for filename in WAV_FILENAMES:
        if filename in except_filenames: # I did those manually, I don't want to lose minutes of the API
            continue
        
        #import pdb; pdb.set_trace() # !! Breakpoint
        with open(ALIGN_DIR + gentle_align_dir + filename[:-4] + '.json', 'r') as f:
            gentle_data = json.load(f)
            gentle_dict = dict(gentle_data)
            
        with open(ALIGN_DIR + mfa_align_dir + filename[:-4] + '.json', 'r') as f:
            mfa_data = json.load(f)
            mfa_dict = dict(mfa_data)
            
        # Merge the two alignments
        gentle_mfa_dict = list()
        for i in range(len(gentle_dict['words'])):
            if 'start' in gentle_dict['words'][i]: # the alignment is in gentle
                gentle_mfa_dict.append(gentle_dict['words'][i])
            else: # the alignment is not in gentle, we use the mfa one
                gentle_dict['words'][i]['start'] = mfa_dict['tiers']['words']['entries'][i][0]
                gentle_dict['words'][i]['end'] = mfa_dict['tiers']['words']['entries'][i][1]
                if i < len(gentle_dict)-1 and 'start' in gentle_dict['words'][i+1]: 
                    # Unless it is a not recognized as well, I also correct the start of the word following the non recognized ones, which might include them
                    gentle_dict['words'][i+1]['start'] = mfa_dict['tiers']['words']['entries'][i+1][0]
                    
                gentle_mfa_dict.append(gentle_dict['words'][i])
        
        # Sort them by the start
        sorted_align = sorted(gentle_mfa_dict, key=lambda d: d['start'])
        
        # Delete inconsistencies
        for i in range(len(sorted_align)):        
            word1 = sorted_align[i]
            if i < len(sorted_align)-1:
                word2 = sorted_align[i+1]
            else:
                break
            
            if word2['start'] < word1['end']: # it's an inconsistency if the start of the second is before the end of the first
                sorted_align[i]['end'] = sorted_align[i+1]['start'] # i reaccord by shifting the end of the first to the start of the second.
            else:
                continue
        
        # Save
        sorted_dict = {'words':sorted_align}
        
        with open(ALIGN_DIR + gentle_mfa_align_dir + 'json/' + filename[:-4] + '.json', 'w') as f:
            json.dump(sorted_dict, f, indent = 1)