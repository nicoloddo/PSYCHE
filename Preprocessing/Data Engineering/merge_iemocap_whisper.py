# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:20:10 2023

@author: nicol


This script merges the Whisper alignment with the IEMOCAP default alignment.
"""
import json
import pandas as pd

import __libpath
from psychelibrary import psyche_dataset as psd
from psychelibrary import nic_dataset_tools as ndt

content = pd.read_pickle("../Save/content.pkl")

psyche = psd.PsycheDataset('IEMOCAP', content)
psyche.set_aligner('whisper')

psyche.parse_alignments('whisper')
psyche.parse_alignments('iemocap_default')

alignments = psyche.alignments

test_convo_alignment = alignments['iemocap_default']['audios']['Ses01F_impro03']
convo_words_clean = [word for word in test_convo_alignment if word['word'] != '<s>' and word['word'] != '</s>']