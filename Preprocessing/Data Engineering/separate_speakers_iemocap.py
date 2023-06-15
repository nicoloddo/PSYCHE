# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 22:18:01 2023

@author: nicol
"""

import pandas as pd
from scipy.io.wavfile import write

import __libpath
from psychelibrary import psyche_dataset as psd


dialog = pd.read_pickle("../Save/dialog.pkl")
subset = dialog.loc[dialog['conversation_id'] == 'Ses01F_impro03']

psyche = psd.PsycheDataset('IEMOCAP', subset)

reduced_noise1, rate1, reduced_noise2, rate2 = psyche.separate_speaker_channels()

write("../Save/test_denoised1.wav", rate1, reduced_noise1)
write("../Save/test_denoised2.wav", rate2, reduced_noise2)