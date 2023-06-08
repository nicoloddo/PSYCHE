# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:58:33 2023

@author: nicol
"""

import json
import pandas as pd

import __libpath
from psychelibrary import psyche_dataset as psd
from psychelibrary import nic_dataset_tools as ndt

time_factor = 1

content = pd.read_pickle("../Save/content.pkl")

psyche = psd.PsycheDataset('IEMOCAP', content)
psyche.set_aligner('whisper')

convo_id = 'Ses01F_impro03'

psyche.parse_alignment_file(convo_id + '.json', 'whisper/')

words = psyche.alignments['whisper/']

turn_words = psyche.parse_alignment_turn_iemocap(ndt.get_iemocap_turnrow(content, convo_id + '_F003'))

convo_words = psyche.parse_alignment_audio(ndt.get_iemocap_turnrow(content, convo_id + '_F003'))
convo_words_clean = [word for word in convo_words if word['word'] != '<s>' and word['word'] != '</s>']

with open('breath_labeling_settings.json', 'r') as f:
    breath_settings = json.load(f)

breath_labeled_convo = []
for word in convo_words_clean:
    if word['word'] == '<sil>': # if it is a silence
        if ndt.breathing_in_word(word, convo_id + '.wav'):
            word['word'] = ndt.BREATH_TOKEN
            breath_labeled_convo.append(word)
        else:
            continue
    else:
        breath_labeled_convo.append(word)
        
breath_labeled_transcr = ' '.join([word['word'] for word in breath_labeled_convo])

with open('../Save/iemocap_alignment_breath_transcript.txt', 'w') as f:
    f.write(breath_labeled_transcr)