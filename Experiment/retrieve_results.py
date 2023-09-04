# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 02:32:11 2023

@author: nicol
"""

import requests
import json

api_key = '$2b$10$N7OfngmWzAbzoEHQGWVbTOaoV.EFbGbqtuxYuFXqO2JOMfNKPk82q'
headers = {'X-Master-key': api_key, 'Content-Type': 'application/json'}

# URL to fetch the uncategorized bins
url = 'https://api.jsonbin.io/v3/c/uncategorized/bins/'

while True:
    # Get uncategorized bins
    response = requests.get(url, headers=headers)
    bins = response.json()

    if 'message' in bins:
        print("Error:", bins['message'])
        break

    if not bins:
        # No more bins to fetch, break the loop
        break

    # Iterate through each bin and print its meta data (or save it as desired)
    for single_bin in bins:
        bin_id = single_bin['record']        
        
        print('Bin ID:', bin_id)
        print('Created At:', single_bin['createdAt'])
        print('Private:', single_bin['private'])
        print('---')
        
        response = requests.get(f'https://api.jsonbin.io/v3/b/{bin_id}', headers=headers)
        with open(f'./Results/{bin_id}.json', 'w') as file:
            json.dump(response.json(), file, indent=3)

    # Get the last bin ID fetched in this batch
    last_bin_id = bins[-1]['record']
    
    # Update the URL to fetch the next set of uncategorized bins
    url = f'https://api.jsonbin.io/v3/c/uncategorized/bins/{last_bin_id}'

print("Finished fetching all uncategorized bins.")

