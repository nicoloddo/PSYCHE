# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:20:10 2023

@author: nicol


This script merges the Whisper alignment with the IEMOCAP default alignment.
"""
import json
import pandas as pd
import string

from common import psyche_dataset as psd
from common import dataset_tools as ndt

import Levenshtein


def overlap(time1, time2, s_range = 1):
    return time1[0] < time2[1] + s_range and time1[1] > time2[0] - s_range


with open('../Save/alignments.json', 'r') as f:
    alignments = json.load(f)

whisper_convo_alignment = alignments['whisper']['audios']['Ses01F_impro03']
#!!! this has to be removed, whisper should transcript the audios already trimmed with the end of the last iemocap sentence transcription
whisper_convo_alignment = [word for word in whisper_convo_alignment if word['end'] < 120]
iemocap_convo_alignment = alignments['iemocap_default']['audios']['Ses01F_impro03']


# The special tags
tags = ['++LAUGHTER++', '++BREATHING++', '++GARBAGE++', '++LIPSMACK++']


# clean iemocap from <s>, </s> and <sil>
iemocap_convo_clean = [word for word in iemocap_convo_alignment if word['word'] != '<s>' and word['word'] != '</s>' and word['word'] != '<sil>']

# Move tags to the whisper alignment
# Collecting indices of special tag segments in alignment2 to remove later
indices_to_remove = []

for i, segment2 in enumerate(iemocap_convo_clean):
    if segment2['word'] in tags:
        # Add a new segment to alignment1 with 'actor' field
        new_segment = segment2.copy()
        new_segment['actor'] = segment2['actor']
        whisper_convo_alignment.append(new_segment)
        
        # Collect the index to remove later
        indices_to_remove.append(i)

        # Check for overlapping segments in alignment1 and adjust their times
        for segment1 in whisper_convo_alignment:
            if overlap([segment1['start'], segment1['end']], [new_segment['start'], new_segment['end']], s_range = 0):
                # If the overlapping segment in alignment1 starts before the new segment, 
                # adjust its end time
                if segment1['start'] < new_segment['start']:
                    segment1['end'] = new_segment['start']
                # If the overlapping segment in alignment1 ends after the new segment,
                # adjust its start time
                elif segment1['end'] > new_segment['end']:
                    segment1['start'] = new_segment['end']

whisper_convo_alignment.sort(key=lambda segment: segment['start'])
# Removing the special tag segments from alignment2
for index in sorted(indices_to_remove, reverse=True):
    del iemocap_convo_clean[index]

whisper_no_tags = [word for word in whisper_convo_alignment if word['word'] not in tags]

# Normalize the words
for word in whisper_no_tags:
    if word['word'] not in tags:
        word['word'] = ndt.normalize_word(word['word'])

for word in iemocap_convo_clean:
    if word['word'] not in tags:
        word['word'] = ndt.normalize_word(word['word'])

# Add `actor` tags to the first alignment
for segment1 in whisper_no_tags:
    # Find overlapping entries in the second alignment
    overlapping_segments = [segment2 for segment2 in iemocap_convo_clean if overlap((segment1['start'], segment1['end']), (segment2['start'], segment2['end']))]
    if not overlapping_segments:
        continue
    # Find the overlapping entry that has the most similar word
    #!!! It is not 100% accurate, the sorting should depend on a weighted value between word similarity and time overlapping similarity
    overlapping_segments.sort(key=lambda segment2: Levenshtein.distance(segment1['word'], segment2['word']))
    # Add the `actor` tag
    segment1['actor'] = overlapping_segments[0]['actor']
