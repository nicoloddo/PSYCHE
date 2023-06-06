# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 19:48:05 2023

@author: nicol
"""
import os
import pandas as pd
import re

ROOT_FOLDER = 'D:\\OneDrive - Universiteit Utrecht\\Documents\\000 - Documenti\\PROJECTS\\PSYCHE\\Dataset\\IEMOCAP_full_release_withoutVideos.tar\\IEMOCAP_full_release_withoutVideos\\IEMOCAP_full_release'

def process_all_files(root_folder):
    dialog_list = []
    content_list = []

    for dir_path, dirs, files in os.walk(root_folder):
        if 'EmoEvaluation' in dir_path:
            # Remove dirs[:] to not visit any sub-directories
            dirs[:] = []
            for file in files:
                if file.endswith('.txt'):  # only process text files
                    full_path = os.path.join(dir_path, file)

                    # append the dialog metadata to dialog_list
                    dialog_metadata = process_file_metadata(full_path, root_folder)
                    dialog_list.append(dialog_metadata)

                    # parse the file content and append to content_list
                    content_data = parse_file_content(full_path, root_folder, dialog_metadata)
                    content_list.append(content_data)

    dialog_df = pd.DataFrame(dialog_list)
    content_df = pd.concat(content_list, ignore_index=True)

    return dialog_df, content_df


def process_file_metadata(file_path, root_folder):
    # extract session, conversation type, conversation number, and leading actor from filename
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # Remove the '.txt' extension
    split = file_name.split('_')
    session_actor = split[0] # before the first _
    convo_info = "_".join([str(item) for item in split[1:]]) # join the rest of the string
    session = session_actor[:5]  # "Ses01"
    session_num = int(session[3:])  # "01" -> 1
    leading_actor_gender = session_actor[5]  # "F" or "M"
    convo_type = convo_info[:-2]  # removes last two digits to get the conversation type
    
    # construct secondary actor
    if leading_actor_gender == 'F':
        secondary_actor_gender = 'M'
    else:
        secondary_actor_gender = 'F'
    secondary_actor = secondary_actor_gender + str(session_num)
    leading_actor = leading_actor_gender + str(session_num)
    # construct conversation id
    convo_id = file_name  # as the convo_id is same as filename (without '.txt')
    
    session_name = 'Session' + str(session_num)
    # construct transcription path
    transcript_path = os.path.join(root_folder, session_name, 'dialog', 'transcriptions', convo_id + '.txt')
    # construct wav path
    wav_path = os.path.join(root_folder, session_name, 'dialog', 'wav', convo_id + '.wav')

    transcripts = {}
    with open(transcript_path, 'r') as f:
        lines = f.read().splitlines()
        lines_clean = [x for x in lines if x.startswith('Ses')] # remove non segmented audios
        for i, line in enumerate(lines_clean):
            turn_id_time, transcription = line.split(':')
            
            if i == 0:
                time = turn_id_time.split('[')[1][:-1].split('-')
                convo_start_time = time[0]
            if i == len(lines_clean) - 1:
                time = turn_id_time.split('[')[1][:-1].split('-')
                convo_end_time = time[1]
                
            turn_id = turn_id_time.split('[')[0].strip()

            transcription = transcription.strip()
            transcripts[turn_id] = transcription
            
    # now return a dict with all these data
    return {
        'start_time' : float(convo_start_time),
        'end_time' : float(convo_end_time),
        'session': session_num,
        'leading_actor': leading_actor,
        'secondary_actor': secondary_actor,
        'conversation_type': convo_type,
        'conversation_id': convo_id,
        'transcription': transcripts,
        'wav_path': wav_path,
    }


def parse_line(line):
    result = re.match(r'\[(\d+\.\d+) - (\d+\.\d+)\]\s+(\S+)\s+(\w+)\s+\[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]', line)
    if result:
        return {
            'start_time': float(result.group(1)),
            'end_time': float(result.group(2)),
            'turn_name': result.group(3),
            'emotion': result.group(4),
            'valence': float(result.group(5)),
            'activation': float(result.group(6)),
            'dominance': float(result.group(7))
        }
    else:
        return None

def parse_file_content(file_path, root_folder, file_metadata):
    with open(file_path, 'r') as f:
        lines = f.readlines()    
    
    transcripts = file_metadata['transcription']
    convo_id = file_metadata['conversation_id']
    data = []
    annotators = []
    labels = []
    other_labels = []
    for line in lines:
        line = line.strip()
        if not line:  # reset on empty lines
            if annotators:
                data[-1]['labelers'] = ';'.join(annotators)
                for i, (label, other_label) in enumerate(zip(labels, other_labels)):
                    if annotators[i] == data[-1]['actor_speaking_id']:
                        data[-1]['self_label'] = label
                        data[-1]['self_other_label'] = other_label
                    else:
                        data[-1]['label_'+str(i+1)] = label
                        data[-1]['other_label_'+str(i+1)] = other_label
            annotators = []
            labels = []
            other_labels = []
            
        elif line.startswith('['):  # new entry
            line_data = parse_line(line)
            if line_data:
                session, leading_actor_gender, convo_type, convo_num, actor_speaking_gender, turn_num = re.match(r'(\S{5})([FM])_(\S+?)(\d{2}(?:_\d)?\w*)_([FM])(\d{2})', line_data['turn_name']).groups()
                    
                session_num = int(session[3:])  # "01" -> 1
                session_folder = 'Session' + str(session_num)
                wav_path = os.path.join(root_folder, session_folder, 'sentences', 'wav', convo_id, line_data['turn_name'] + '.wav')
                alignment_path = os.path.join(root_folder, session_folder, 'sentences', 'ForcedAlignment', convo_id, line_data['turn_name'] + '.wdseg')
                
                line_data['session'] = session_num
                line_data['conversation_id'] = convo_id
                line_data['conversation_type'] = convo_type
                line_data['leading_dialog_actor'] = leading_actor_gender + str(line_data['session'])
                line_data['secondary_dialog_actor'] = ('M' if leading_actor_gender == 'F' else 'F') + str(line_data['session'])
                line_data['actor_speaking_id'] = actor_speaking_gender + str(line_data['session'])
                line_data['transcription'] = transcripts[line_data['turn_name']]
                line_data['wav_path'] = wav_path
                line_data['alignment_path'] = alignment_path
                
                data.append(line_data)
                
                
                
        elif line.startswith('C-'):  # annotator line
            parts = line.split(':')
            annotator = parts[0][2:]
            annotators.append(annotator)
            label, other_label = re.match(r'(.+);\s+\((.*)\)', parts[1]).groups()
            labels.append(label.strip())
            other_labels.append(other_label)

    return pd.DataFrame(data)


dialog_df, content_df = process_all_files(ROOT_FOLDER)

dialog_df['conversation_duration'] = dialog_df['end_time'].sub(dialog_df['start_time'], axis = 0)
content_df['turn_duration'] = content_df['end_time'].sub(content_df['start_time'], axis = 0)

dialog_df.to_pickle("Save/dialog.pkl")
content_df.to_pickle("Save/content.pkl")
