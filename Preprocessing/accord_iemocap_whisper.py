# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 21:03:43 2023

@author: nicol
"""

import re
import nltk
import numpy as np
from nltk.metrics.distance import edit_distance
nltk.download('punkt')

# Transcription 1 with 'M:' and 'F:' labels
transcript1 = '''M: So what's up?  What's new?
F: Well Vegas was awesome.
M: Yeah.  I heard.
F: And, um, I got married.
M: Shut up.  No- in Vegas?
F: Yeah.  In the old town part.
M: Who did you marry?
F: Chuck. [LAUGHTER]
M: Did you propose to you?
F: Um- Yes.  It was very romantic.
F: It was at the slot machines.
M: Oh.  The Wheel of Fortune slots?
F: Uhuh, uhuh.  He won big and he-and he realized that the only thing that would make it better was me as his bride.
M: He turned to you and was like...
F: Hey.  Let's get married and I said okay.
M: [LAUGHTER] That's really romantic.
F: Yeah.  Well.  You know, because he's leaving the next day.
M: Yeah.
F: But we're going to have a honeymoon cruise.
M: Does that mean that you're going to get citizenship, too, in- in England or whatever?
F: Oh.  I hadn't even thought about that.
M: Yeah.  Think about that.
F: He's not going to be a citizen, though.
M: Yeah.  But he'll have like a long visa.  Can you go visit then for a long time?
M: I think you can.
F: Oh.  Totally.
M: Yeah. 
M: Oh, how are you going to do the long distance thing?  So wait, are you going to move there?
F: I guess I'll-  I guess I'll have an internet husband.[LAUGHTER]
M: Like you need another one of those.[LAUGHTER]
F: I have an internet boyfriend.  I guess I'll have to juggle the two.  Is that cheating?  I don't know. Hmm.
M: Is that your phone?
F: Yeah, so [LAUGHTER] He's calling me now
M: He loves you.
F: He likes to keep tabs, you know.
M: How much did he win at the slot?
F: Uh, I think seven hundred and fifty dollars.
M: Really? Penny slots?
F: Penny slots. That's what he plays.
M: Wow.
F: Yeah.
M: I'm a big fan of the Wheel of Fortune quarters. But it just costs so much.  I don't know.
F: Oh.
F: Yeah.  We played Wheel of Fortune pennies.  It's like a giant thing with like-
M: But the pennies always get you because you end up spending like, you know, fifty bucks.
F: Fifty bucks. Mmhmm.
M: like, wait, but it's just pennies.  You're like, wait a minute I'm on to you.
F: Yeah.
F: Well so now we're married. We have cat children.
M: Awesome.
M: [LAUGHTER] Cat babies.
F: We renamed Brenda, Lumber Janet.
M: That's fair enough.
F: She needs a new name each state that we go into. [LAUGHTER]
'''

# Transcription 2 without labels
transcript2 = '''So what's up? What's new? Well, Vegas was awesome. Yeah, I heard. And, uh, I got married. Shut up. In Vegas? Yeah, in the old town part. Who'd you marry? Chuck! Did he propose to you? Um, yes. It was very romantic. It was at the slot machines. Oh, Wheel of Fortune slots? Uh-huh. He won big and he realized that the only thing that would make it better with me is his bride. He turned to you and was like... Hey, let's get married. It's really romantic. Yeah, well, you know, because he's leaving the next day. Yeah. But we're going to have a honeymoon cruise. Does that mean you're going to get citizenship too in England or whatever? Oh, I hadn't even thought about that. Yeah, think about that. He's not going to be a citizen though. Yeah, but he'll have like a long visa. Can you go visit them for a long time? Oh, totally. Oh, how are you going to do the long distancing? So wait, are you going to move there? I guess I'll have an internet husband. You think you need another one of those? I have an internet boyfriend. I guess I'll have to juggle the two. Is that cheating? Yeah. Is that your phone? No, we're kidding. He's calling me now. He loves you. He likes to keep tabs, you know. How much did he win on the slot? I think $750. Really? Penny slots. Penny slots? That's what he plays. Wow. Yeah. I'm a big fan of the Wheel of Fortune quarters. Oh. But it just costs so much. Yeah. We played Wheel of Fortune pennies. It's like a giant thing. But the pennies always get you because then you end up spending like, you know. 50 bucks. 50 bucks. You're like, wait, but it's just pennies. But so, now we're married. Awesome. We have cat children. Cat babies. We renamed Brenda Lumber Janet. She needs a new name each state that we go into.
'''

# Remove the square brackets words first of all 
#!!! implement something to put them back as difluency tags
transcript1 = re.sub(r'\[.*?\]', '', transcript1)
# Tokenize both transcriptions into sentences
sentences1 = nltk.sent_tokenize(transcript1)
sentences2 = nltk.sent_tokenize(transcript2)

# Initialize list to store labeled sentences for transcription 2
labeled_sentences2 = []

# Initialize label variable to hold the most recent speaker label
label = None


best_i = 0
max_iterator = len(sentences1)
min_iterator = 0
iterator_sentences_range = 3
# Iterate over sentences in transcription 2
for sentence2 in sentences2:
    # Initialize minimum edit distance to a high value
    min_distance = float('inf')
    # Initialize best match sentence
    best_match = None
    
    # Iterate over sentences in transcription 1
    bounds = [(best_i+1) - iterator_sentences_range, (best_i+1) + iterator_sentences_range]
    bounds = np.clip(bounds, min_iterator, max_iterator)
    for i in range(bounds[0], bounds[1]):
        sentence1 = sentences1[i]
        # Update label if this sentence has one
        if 'M:' in sentences1[i] or 'F:' in sentences1[i]:
            label = re.search(r'[\r\n\s]*[MF]:', sentences1[i]).group().strip()
        # Compute Levenshtein distance between current sentence in transcription 1 and 2, removing the label
        distance = edit_distance(re.sub('^[MF]: ', '', sentences1[i]), sentence2)
        # If this distance is the smallest so far, update minimum distance and best match sentence
        if distance < min_distance:
            min_distance = distance
            best_match = sentences1[i]
            best_label = label
            best_i = i
    # Add the label to the sentence from transcription 2 and append it to the list of labeled sentences
    labeled_sentences2.append(best_label + ' ' + sentence2)

# Join labeled sentences into a single string
labeled_transcript2 = '\n'.join(labeled_sentences2)

print(labeled_transcript2)