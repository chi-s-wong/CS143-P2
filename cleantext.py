#!/usr/bin/env python

"""Clean comment text for easier parsing."""

from __future__ import print_function
import unittest
import re
import string
import argparse
import json

__author__ = ""
__email__ = ""

# Depending on your implementation,
# this data may or may not be useful.
# Many students last year found it redundant.
_CONTRACTIONS = {
    "tis": "'tis",
    "aint": "ain't",
    "amnt": "amn't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hell": "he'll",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "ol": "'ol",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "somebodys": "somebody's",
    "someones": "someone's",
    "somethings": "something's",
    "thatll": "that'll",
    "thats": "that's",
    "thatd": "that'd",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "wedve": "wed've",
    "well": "we'll",
    "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatd": "what'd",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whodve": "whod've",
    "wholl": "who'll",
    "whore": "who're",
    "whos": "who's",
    "whove": "who've",
    "whyd": "why'd",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've"
}

# You may need to write regular expressions.

def sanitize(text):
    """Do parse the text in variable "text" according to the spec, and return
    a LIST containing FOUR strings 
    1. The parsed text.
    2. The unigrams
    3. The bigrams
    4. The trigrams
    """
    
    # Remove all non-space whitespace
    text = text.lower()
    text = re.sub('\s+',' ',text)
    # Remove URLs
    text = re.sub('(http[s]?://)?www.\S+', '', text)    
    # Remove links to subreddits and users
    text = re.sub('\/r\/[_\-a-z0-9A-Z]*', '', text)
    text = re.sub('\/u\/[_\-a-z0-9A-Z]*', '', text)
    # Remove uncommon puncuation
    text = re.sub('(?<![a-z])[\[\]"*&()%$@]+(?=[a-z])', '', text)
    text = re.sub('(?<=[a-z])[\[\]"*&()%$@]+(?![a-z])', '', text)
    # Pad Punctuation
    text = re.sub('([.,!?;:])(?![a-z])', r' \1 ', text)
    # Split text on a single space.
    text = text.split()
    parsed_text = ""
    unigrams = ""
    bigrams = ""
    trigrams = ""

    for token in text:
    	parsed_text += token + ' '
    print('Parsed Text:\n'+ parsed_text + '\n--------\n')

    # Separate all external punctuation such as periods, commas, etc. into their own tokens (a token is a single piece of text with no spaces), but maintain punctuation within words
    return [parsed_text, unigrams, bigrams, trigrams]




if __name__ == "__main__":
    # This is the Python main function.
    # You should be able to run
    # python cleantext.py <filename>
    # and this "main" function will open the file,
    # read it line by line, extract the proper value from the JSON,
    # pass to "sanitize" and print the result as a list.
    
    # YOUR CODE GOES BELOW.

    # Grab filename from user input
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    filename = parser.parse_args().filename
    # Extract comment from file line by line
    with open(filename) as f:
#        for line in f:
        for i, line in enumerate(f):
            if i == 4:
                line = f.readline()
                data_dict = json.loads(line)
                print(data_dict['body'])
                print(sanitize(data_dict['body']))

