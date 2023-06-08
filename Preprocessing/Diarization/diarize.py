# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:20:10 2023

@author: nicol
"""
import json
import pandas as pd

import __libpath
from psychelibrary import psyche_dataset as psd
from psychelibrary import nic_dataset_tools as ndt

content = pd.read_pickle("../Save/content.pkl")

psyche = psd.PsycheDataset('IEMOCAP', content)
psyche.set_aligner('whisper')

psyche.parse_alignment_file('Ses01F_impro03.json', 'whisper/')

words = psyche.alignments['whisper/']

turn_words = psyche.parse_alignment_turn_iemocap(ndt.get_iemocap_turnrow(content, 'Ses01F_impro03_F003'))

convo_words = psyche.parse_alignment_audio(ndt.get_iemocap_turnrow(content, 'Ses01F_impro03_F003'))
convo_words_clean = [word for word in convo_words if word['word'] != '<s>' and word['word'] != '</s>']