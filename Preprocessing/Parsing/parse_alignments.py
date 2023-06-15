# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 13:48:24 2023

@author: nicol

This script parses the alignments and saves them for easy retrieval.
"""

import json
import pandas as pd

import __libpath
from psychelibrary import psyche_dataset as psd
from psychelibrary import nic_dataset_tools as ndt
_, _, ALIGN_DIR, _, _, _, _ = ndt.setup_global_paths() # Here it uses the default ones

already_parsed = True

if not already_parsed:
    content = pd.read_pickle("../Save/content.pkl")
    
    psyche = psd.PsycheDataset('IEMOCAP', content)
    psyche.set_aligner('whisper')
    
    psyche.parse_alignments('whisper')
    psyche.parse_alignments('iemocap_default')
    
    alignments = psyche.alignments
    
    with open('../Save/alignments.json', 'w+') as f:
        json.dump(alignments, f, indent = 6)
else:
    with open('../Save/alignments.json', 'r') as f:
        alignments = json.load(f)
    
export_iemocap_aligments = True
if export_iemocap_aligments:
    ndt.ALIGNER = 'iemocap_default'
    ndt.DATASET = 'IEMOCAP'
    
    
    full_save_path = ALIGN_DIR + 'iemocap_default/' + 'json/'
    
    iemocap_alignments = alignments['iemocap_default']['audios']    
    
    for convo_id in iemocap_alignments:
        # remove <sil>, <s> and </s>
        iemocap_alignments[convo_id] = [word for word in iemocap_alignments[convo_id] if word['word'] != '<sil>' and word['word'] != '<s>' and word['word'] != '</s>']
        
        # report the tags as <tag>, lower the characters and remove word versions '(2)' and punctuation
        tags = ['++LAUGHTER++', '++BREATHING++', '++GARBAGE++', '++LIPSMACK++']
        for word in iemocap_alignments[convo_id]:
            if word['word'] == '<breath>':
                continue
            elif word['word'] not in tags:
                word['word'] = ndt.normalize_word(word['word'])
            else:
                tag = ndt.normalize_word(word['word'])
                word['word'] = '<' + tag + '>'
        
        # save
        with open(full_save_path + convo_id + '.json', 'w') as f:
            json.dump({'words' : iemocap_alignments[convo_id]}, f, indent = 1)
        