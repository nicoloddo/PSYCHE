# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 17:57:38 2023

@author: nicol
"""

import random

def split_file(filelist_name, train_percentage):
    input_file = filelist_name + '.txt'
    train_file = filelist_name + '_train.txt'
    valid_file = filelist_name + '_val.txt'
    
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Shuffle lines
    random.shuffle(lines)

    # Compute split index
    split_idx = int(len(lines) * train_percentage)

    # Split lines into two lists
    train_lines = lines[:split_idx]
    valid_lines = lines[split_idx:]

    # Write to train file
    with open(train_file, 'w') as f:
        f.writelines(train_lines)

    # Write to valid file
    with open(valid_file, 'w') as f:
        f.writelines(valid_lines)

    print(f"Total number of rows: {len(lines)}")
    print(f"Number of training rows: {len(train_lines)}")
    print(f"Number of validation rows: {len(valid_lines)}")

# Call the function
split_file('IEMOCAP_INTERSPEECH_filelist', 0.9)
