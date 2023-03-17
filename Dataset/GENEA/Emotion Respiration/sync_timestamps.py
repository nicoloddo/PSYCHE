
#%%
import ffmpeg
import sys
import datetime
import time
import glob
import pandas as pd
import opensmile
import numpy as np
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01b,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

list_files = glob.glob('video_data/*.MP4')
df_exists = False
for file in list_files:
    # read the audio/video file from the command line arguments
    # uses ffprobe command to extract all possible metadata from the media file

    probe = ffmpeg.probe(file)["streams"][0]
    creation_time = probe['tags']['creation_time']
    frame_rate = probe['r_frame_rate']
    pt = time.strptime(creation_time,'%Y-%m-%dT%H:%M:%S.%fZ')

    row = {'file_path': file, 'date': '%.2d-%.2d-%s' % (pt.tm_mday, pt.tm_mon, pt.tm_year), 
    'time - NL': '%.2d:%.2d:%.2d' % (pt.tm_hour + 1, pt.tm_min, pt.tm_sec), 
    'time - UTC': '%.2d:%.2d:%.2d' % (pt.tm_hour, pt.tm_min, pt.tm_sec), 
    'sample_rate' : frame_rate}

    if not df_exists:
        df = pd.DataFrame(row, index = [0])
        df_exists = True
    else:
        df = df.append(pd.DataFrame(row, index = [0]), ignore_index= True)

    #y = smile.process_file(file)
    #y.to_csv(file.replace('.MP4', '_LLD.csv'))
    #ssss
df.to_csv('timestamps_videos.csv')

# %%
list_files = glob.glob('interface_data/*.csv')
correspondence_videos = pd.read_excel('video_correspondence.xlsx')
sam_scores = pd.read_excel('SAM_data_allparticipants.xlsx',sheet_name= 'allparticipants')
sam_scores['id'] = sam_scores['id'].map(str)

df_exists = False
for file in list_files:
    interface_info = pd.read_csv(file, sep=';')
    
    
    id = file.split('/')[1].split('_')[0]
    participant_scores = sam_scores[sam_scores['id'] == id]

    video = correspondence_videos[correspondence_videos['ID'] == id]['video'].values


    pt = time.strptime(interface_info['date'].values[0],'%Y_%B_%d_%H%M')
    pt_1 = datetime.datetime.strptime(interface_info['date'].values[0],'%Y_%B_%d_%H%M')
    
    start_record_unix = datetime.datetime.timestamp(pt_1) #- 60*60*2

    if 'baseline_time' in interface_info.columns or 'wrong_baseline_time' in interface_info.columns:
        if 'baseline_time' in interface_info.columns:
            baseline_start_unix = np.nansum(interface_info['baseline_time'].values)
            baseline_start = datetime.datetime.utcfromtimestamp(baseline_start_unix).strftime('%d-%m-%Y %H:%M:%S')
            baseline_stop_unix =  baseline_start_unix + 45
            baseline_stop = datetime.datetime.utcfromtimestamp(baseline_stop_unix).strftime('%d-%m-%Y %H:%M:%S')

        # wrong baseline time - the baseline time here is actually the start of the calibration time
        if 'wrong_baseline_time' in interface_info.columns:
            baseline_start_unix = start_record_unix + float(interface_info[interface_info['eyesclosed2.started'].notnull()]['eyesclosed2.started'].values[0].replace(',', '.'))
            baseline_start = datetime.datetime.utcfromtimestamp(baseline_start_unix).strftime('%d-%m-%Y %H:%M:%S')
            baseline_stop_unix =  baseline_start_unix + 45
            baseline_stop = datetime.datetime.utcfromtimestamp(baseline_stop_unix).strftime('%d-%m-%Y %H:%M:%S')
            
        row = {'id': file.split('/')[1].split('_')[0], 'video': video, 'file_path': file, 'date': '%.2d-%.2d-%s' % (pt.tm_mday, pt.tm_mon, pt.tm_year), 
        'time - NL': '%.2d:%.2d:%.2d' % (pt.tm_hour, pt.tm_min, pt.tm_sec), 
        'time - UTC': '%.2d:%.2d:%.2d' % (pt.tm_hour - 2, pt.tm_min, pt.tm_sec), 
        'baseline_start' :  baseline_start.split(' ')[-1],
        'baseline_start_unix' :  baseline_start_unix,
        'baseline_stop' : baseline_stop.split(' ')[-1],
        'baseline_stop_unix' :  baseline_start_unix,
        }

        row['V_baseline'] = participant_scores['V_baseline'].values
        row['A_baseline'] = participant_scores['A_baseline'].values

        for index, song in interface_info.iloc[np.shape(interface_info)[0] - 4:].iterrows():
            index = index -  (np.shape(interface_info)[0] - 4)
            name = song['song_file'].split('/')[1].split('.')[0] 
            
            start_song_unix = song['stimulus_scrn_stime']
            start_song = datetime.datetime.utcfromtimestamp(start_song_unix).strftime('%d-%m-%Y %H:%M:%S')
            stop_song_unix = start_song_unix + 90
            stop_song = datetime.datetime.utcfromtimestamp(stop_song_unix).strftime('%d-%m-%Y %H:%M:%S')
            
            start_reflection_unix = song['reflection_time_start']
            start_reflection = datetime.datetime.utcfromtimestamp(start_reflection_unix).strftime('%d-%m-%Y %H:%M:%S')
            stop_reflection_unix =  start_record_unix + float(song['text_13.stopped'].replace(',','.'))
            stop_reflection = datetime.datetime.utcfromtimestamp(stop_reflection_unix).strftime('%d-%m-%Y %H:%M:%S')
            
            row['%d_start_song_unix' % index] = start_song_unix
            row['%d_start_song' % index] = start_song.split(' ')[-1]
            row['%d_stop_song_unix' % index] = stop_song_unix
            row['%d_stop_song' % index] = stop_song.split(' ')[-1]

            row['%d_start_reflection_unix' % index] = start_reflection_unix
            row['%d_start_reflection' % index] = start_reflection.split(' ')[-1]
            row['%d_stop_reflection_unix' % index] = stop_reflection_unix
            row['%d_stop_reflection' % index] = stop_reflection.split(' ')[-1]
            
            row['%d_song' % index] = name
            row['V_%d_song' % index] = participant_scores['V_%s' % name].values
            row['A_%d_song' % index] = participant_scores['A_%s' % name].values
        #background4.started
        #row = {'file_path' : file, 'date': 
        #'start time - NL':
        #}
        if not df_exists:
            df = pd.DataFrame(row, index = [0])
            df_exists = True
        else:
            df = df.append(pd.DataFrame(row, index = [0]), ignore_index= True)

    df.to_csv('timestamps_experiments.csv')

for index, c in correspondence_videos.iterrows():
    if c['video'] not in df['video'].values:
        print('index - %d, video: %s id: %s' % (index, c['video'], c['ID']))
        # then add the start and stop of each stimulus
    # %%
