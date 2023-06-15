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

                    # append the dialog to dialog_list
                    dialog = process_file(full_path, root_folder)
                    dialog_list.append(dialog)

                    # parse the file content and append to content_list
                    emotional_content_data = parse_emotional_content(full_path, dialog)
                    content_list.append(emotional_content_data)

    dialog_df = pd.DataFrame(dialog_list)
    content_df = pd.concat(content_list, ignore_index=True)

    return dialog_df, content_df


def process_file(file_path, root_folder):
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
    
    turns_data, convo_start_time, convo_end_time, transcription = parse_transcription_file(transcript_path, convo_id, root_folder)
    # now return a dict with all these data
    return {
        'start_time' : float(convo_start_time),
        'end_time' : float(convo_end_time),
        'session': session_num,
        'leading_actor': leading_actor,
        'secondary_actor': secondary_actor,
        'conversation_type': convo_type,
        'conversation_id': convo_id,
        'transcription': transcription,
        'turns_data': turns_data,
        'wav_path': wav_path,
    }

def parse_transcription_file(file_path, convo_id, root_folder):
    turns_data = {}
    transcr_lines = []
    start_time = None
    end_time = None
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # Check if the line matches the pattern "Ses01F_impro03_M002 [010.5300-013.8400]: Shut up.  No- in Vegas?"
            match = re.match(r'(\S+)\s+\[(\d+\.\d+)-(\d+\.\d+)\]:\s+(.*)', line)
            
            if match:
                turn_name = match.group(1)
                turn_pattern_match = re.match(r'(\S{5})([FM])_(\S+?)(\d{2}(?:_\d)?\w*)_([FM])(\d{2})', turn_name)
                
                if turn_pattern_match:
                    turn_transcript = match.group(4)
                    current_start_time = match.group(2)
                    current_end_time = match.group(3)
                    session, leading_actor_gender, convo_type, convo_num, actor_speaking_gender, turn_num = re.match(r'(\S{5})([FM])_(\S+?)(\d{2}(?:_\d)?\w*)_([FM])(\d{2})', turn_name).groups()
                    
                    # save the turn
                    turn_entry = parse_turn(turn_name, convo_id, root_folder, turn_transcript)
                    turns_data[turn_name] = turn_entry
                    
                    # update the start and end time of entire conversation if needed
                    start_time = current_start_time if start_time is None else min(start_time, current_start_time)
                    end_time = current_end_time if end_time is None else max(end_time, current_end_time)
                    
                    # keep only the gender of the actor inside the transcription
                    transcr_line = turn_entry['actor_speaking_id'][:-1] + ': ' + turn_transcript
                    transcr_lines.append(transcr_line)
                else:
                    actor_speaking_gender, _ = re.match(r'\S{5}[FM]_\S+?\d{2}(?:_\d)?\w*_([FM])(\w*)', turn_name).groups()
                    transcr_line = actor_speaking_gender + ': ' + turn_transcript
                    transcr_lines.append(transcr_line)
            else:
                transcr_lines.append(line)

    return turns_data, start_time, end_time, '\n'.join(transcr_lines)

def parse_turn(turn_name, convo_id, root_folder, turn_transcript):
    turn_entry = {}
    
    session, leading_actor_gender, convo_type, convo_num, actor_speaking_gender, turn_num = re.match(r'(\S{5})([FM])_(\S+?)(\d{2}(?:_\d)?\w*)_([FM])(\d{2})', turn_name).groups()
        
    session_num = int(session[3:])  # "01" -> 1
    session_folder = 'Session' + str(session_num)
    wav_path = os.path.join(root_folder, session_folder, 'sentences', 'wav', convo_id, turn_name + '.wav')
    alignment_path = os.path.join(root_folder, session_folder, 'sentences', 'ForcedAlignment', convo_id, turn_name + '.wdseg')
    
    turn_entry['session'] = session_num
    turn_entry['conversation_id'] = convo_id
    turn_entry['conversation_type'] = convo_type
    turn_entry['leading_dialog_actor'] = leading_actor_gender + str(turn_entry['session'])
    turn_entry['secondary_dialog_actor'] = ('M' if leading_actor_gender == 'F' else 'F') + str(turn_entry['session'])
    turn_entry['actor_speaking_id'] = actor_speaking_gender + str(turn_entry['session'])
    turn_entry['transcription'] = turn_transcript 
    turn_entry['wav_path'] = wav_path
    turn_entry['alignment_path'] = alignment_path
    
    return turn_entry

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

def parse_emotional_content(file_path, dialog):
    with open(file_path, 'r') as f:
        lines = f.readlines()    
    
    turns_data = dialog['turns_data']
    #convo_id = file_metadata['conversation_id']
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
                line_data.update(turns_data[line_data['turn_name']])
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

dialog_df['duration'] = dialog_df['end_time'].sub(dialog_df['start_time'], axis = 0)
content_df['duration'] = content_df['end_time'].sub(content_df['start_time'], axis = 0)

# Ensure that 'conversation_id' is the index in the dialog_df
dialog_df = dialog_df.set_index('conversation_id')

# Rename the 'wav_path' column to 'conversation_wav_path' in dialog_df
dialog_df = dialog_df.rename(columns={'wav_path': 'conversation_wav_path'})

# Perform the merge operation with left join to keep every row from content_df
merged_df = content_df.merge(dialog_df[['conversation_wav_path']], left_on='conversation_id', right_index=True, how='left')

save = True
if save:
    dialog_df.to_pickle("../Save/IEMOCAP/dialog.pkl")
    merged_df.to_pickle("../Save/IEMOCAP/content.pkl")
