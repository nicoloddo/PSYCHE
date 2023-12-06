# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:58:33 2023

@author: nicol
"""

import json
import pandas as pd

from common import psyche_dataset as psd
from common import dataset_tools as ndt
ndt.ALIGNER = 'iemocap_default'

time_factor = 1

with open('../Save/alignments.json', 'r') as f:
    alignments = json.load(f)

convo_id = 'Ses01F_impro03'

convo_words = alignments['iemocap_default']['audios'][convo_id]
convo_words_clean = [word for word in convo_words if word['word'] != '<s>' and word['word'] != '</s>']

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

tags = ['++LAUGHTER++', '++BREATHING++', '++GARBAGE++', '++LIPSMACK++']
for word in breath_labeled_convo:
    if word['word'] == ndt.BREATH_TOKEN:
        continue
    elif word['word'] not in tags:
        word['word'] = ndt.normalize_word(word['word'])
    else:
        tag = ndt.normalize_word(word['word'])
        word['word'] = '{' + tag + '}'
        
breath_labeled_transcr = ' '.join([word['word'] for word in breath_labeled_convo])

with open('../Save/iemocap_alignment_sil_breath_transcript.txt', 'w') as f:
    f.write(breath_labeled_transcr)