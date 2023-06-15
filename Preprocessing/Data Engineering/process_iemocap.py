# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:28:49 2023

@author: nicol
"""

import pandas as pd
import re

content = pd.read_pickle("../Save/IEMOCAP/content.pkl")

# Let's create a new emotion label column without the undecided 'xxx' values.
content['emotion_label'] = content['emotion']

# Function to process the labels
def process_labels(row):
    '''

    Parameters
    ----------
    row : THE ROW OF THE DATAFRAME

    Returns
    -------
    Returns the new decided emotion label.
    The label is decided by checking the self label of the actor and selecting the least common emotion among the ones selected by them.
    We select the least common to balance better the dataset. We exclude the 'other' label of emotion because it is not a useful label.

    '''
    
    if row['emotion_label'] == 'xxx':
        self_label_exist_and_is_not_only_other = pd.notnull(row['self_label']) and re.sub(r'[^A-Za-z0-9]+', '', row['self_label']) != 'Other' # the regex removes spaces and punctuation.
        
        if self_label_exist_and_is_not_only_other: # If the self label is not null and not only saying 'Other', we follow the self suggestion
            # Split the self_label, lowercase, and take first three letters
            labels = [label.lower().strip()[:3] for label in row['self_label'].split(';') if label.lower().strip()[:3] != 'oth']
        elif pd.notnull(row['label_1']) and pd.notnull(row['label_2']) and pd.notnull(row['label_3']): # else, we follow the labelers
            # Concatenate and process the labels from label_1, label_2, and label_3
            labels = [label.lower().strip()[:3] for label in (row['label_1'] + ';' + row['label_2'] + ';' + row['label_3']).split(';') if label.lower().strip()[:3] != 'oth']
        else:
            labels = ['xxx']

        if labels:
            label_counts = content['emotion_label'].value_counts()
            least_common_label = min(labels, key=lambda label: label_counts.get(label, 0))
            if least_common_label.strip() == '':
                print('ciao')
            return least_common_label
        else:
            return 'xxx'
    else:
        return row['emotion_label']

# Apply the function to each row
content['emotion_label'] = content.apply(process_labels, axis=1)

print("Removed ambiguities.")
print(content['emotion_label'].value_counts())

# Remove 'Other' emotions which are only three anyway.
dataset = content.loc[content['emotion_label'] != 'oth']

save = True
if save:
    dataset.to_pickle("../Save/IEMOCAP/content_processed.pkl")
    
    