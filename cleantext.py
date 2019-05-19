#!/usr/bin/env python

"""Clean comment text for easier parsing."""

from __future__ import print_function
import unittest
import re
import string
import argparse
import sys

import json

__author__ = ""
__email__ = ""


common = ["!", ".", ",", "?",";", ":"]


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
    # print('Input Text:\n' + text + '\n\n')
    text = re.sub(r'\s+',' ',text)
    # Remove URLs
    text = re.sub(r'((http[s]?://)?www.\S+)|(http[s]?://\S+)', '', text)   
    # text = re.sub(r'\[(.*)\]\(([\/u\/\S+]+)\)', r'\1', text)
    # text = re.sub(r'\[(.*)\]\(([\/r\/\S+]+)\)', r'\1', text)
    text = re.sub(r'(\[.*\])(\(.*\))', r'\1', text)
    
    # # Remove links to subreddits and users
    # text = re.sub('\/r\/[_\-a-z0-9A-Z]*', '', text)
    # text = re.sub('\/u\/[_\-a-z0-9A-Z]*', '', text)
    words = text.split()
    tokens = []
    for word in words:
    	separate_tokens(word, tokens)
    # print("After cleaning")
    # print(tokens)
    # print('\n')

    # Pad Punctuation
    # Split text on a single space.
    parsed_text = ""
    unigrams = ""
    bigrams = ""
    trigrams = ""
    for index, token in enumerate(tokens):
    	parsed_text += token + ' '
    	if token not in common:
    		unigrams += token + ' '
    		if index + 1 <= len(tokens)-1 and tokens[index+1] not in common:
    			bigram = token + '_' + tokens[index+1]
    			bigrams +=  bigram + ' '
    			if index + 2 <= len(tokens)-1 and tokens[index+2] not in common:
    				trigrams += bigram + '_' + tokens[index+2] + ' '
    
    # print('Parsed Text:\n'+ parsed_text + '\n')
    # print('Unigrams:\n'+ unigrams + '\n')
    # print('Bigrams:\n'+ bigrams +'\n')
    # print('Trigrams:\n'+ trigrams + '\n')

    return [parsed_text.strip(), unigrams.strip(), bigrams.strip(), trigrams.strip()]


def separate_tokens(word, token_list):
	# if this word begins and ends with letters or numbers, we dont
	# have any work to do, just add it to the token_list and return
	# Ex's: "aaaaa", "a!!!7"
	if both_alnum(word):
		token_list.append(word)
		return
	# For each token, record the index of the first letter and last letter
	# EX: "<<<a!!m" --> first_letter = 3, last_letter = 6
	first_letter, last_letter = get_indices(word)

	# Keep track of a word to hold embedded punctuation
	cur_word = ""
	# Keep a temporary list of the non-embedded punctuation we see at the end of a token:
	# EX: aaa!!! because we want to form the word before appending these tokens
	later_tokens = []
	for index, letter in enumerate(word):
		# if its a common punc at the beginning of a token, just add it to our token_list immediately
		if letter in common:
			if index < first_letter:
				token_list.append(letter)
			# if its a common punc at the middle of a token between letters, add it to our current word
			elif index > first_letter and index < last_letter:
				cur_word += letter
			# Append to later_tokens to add after this overall token has been processed
			else:
				later_tokens.append(letter)
		# For letters, "'", or special punctuation in between letters
		elif letter.isalnum() or letter == "'" or (index > first_letter and index < last_letter):
			cur_word += letter
	# Only add > 0 length words
	if cur_word != "":
		token_list.append(cur_word)
	# Append all the end-of-token punc marks
	for token in later_tokens:	
		token_list.append(token)


def get_indices(word):
	first_letter = -1
	last_letter = -1
	for index, letter in enumerate(word):
		if letter.isalnum():
			if first_letter == -1:
				first_letter = index
			last_letter = index
	return first_letter, last_letter

def both_alnum(word):
	return True if word[0].isalnum() and word[len(word)-1].isalnum() else False

class TestItems(unittest.TestCase):

	def test_leading_punctuation(self):
		res = sanitize("!!!!meow")
		self.assertEqual(res[0], "! ! ! ! meow")
		self.assertEqual(res[1], "meow")
		self.assertEqual(res[2], "")
		self.assertEqual(res[3], "")

	def test_lots_of_whitespace(self):
		res = sanitize("     hey my name        is chi     ")
		self.assertEqual(res[0], "hey my name is chi")
		self.assertEqual(res[1], "hey my name is chi")
		self.assertEqual(res[2], "hey_my my_name name_is is_chi")
		self.assertEqual(res[3], "hey_my_name my_name_is name_is_chi")

	def test_trailing_punctuation(self):
		res = sanitize("meow!!!!")
		self.assertEqual(res[0], "meow ! ! ! !")
		self.assertEqual(res[1], "meow")
		self.assertEqual(res[2], "")
		self.assertEqual(res[3], "")

	def test_new_line_chars(self):
		res = sanitize("wow\nthis\nlooks\nreally\tcool\njoinme?")
		self.assertEqual(res[0], "wow this looks really cool joinme ?")
		self.assertEqual(res[1], "wow this looks really cool joinme")
		self.assertEqual(res[2], "wow_this this_looks looks_really really_cool cool_joinme")
		self.assertEqual(res[3], "wow_this_looks this_looks_really looks_really_cool really_cool_joinme")

	def test_random_string(self):
		res = sanitize("what>>me!!! whew, i looked at THAT! cat,\" that,\" was, really far ****away***")
		self.assertEqual(res[0], "what>>me ! ! ! whew , i looked at that ! cat , that , was , really far away")
		self.assertEqual(res[1], "what>>me whew i looked at that cat that was really far away")
		self.assertEqual(res[2], "i_looked looked_at at_that really_far far_away")
		self.assertEqual(res[3], "i_looked_at looked_at_that really_far_away")

	def test_embedded_punc(self):
		res = sanitize("wh&&&&at***s")
		self.assertEqual(res[0], "wh&&&&at***s")
		self.assertEqual(res[1], "wh&&&&at***s")
		self.assertEqual(res[2], "")
		self.assertEqual(res[3], "")

	def test_embedded_punc_mixed(self):
		res = sanitize("what**!!..me!!! wow ...youare..mm really cool")
		self.assertEqual(res[0], "what**!!..me ! ! ! wow . . . youare..mm really cool")
		self.assertEqual(res[1], "what**!!..me wow youare..mm really cool")
		self.assertEqual(res[2], "youare..mm_really really_cool")
		self.assertEqual(res[3], "youare..mm_really_cool")


	def test_example_from_spec(self):
		res = sanitize("I'm afraid I can't- explain myself, sir. Because I am not myself, you see?")
		self.assertEqual(res[0], "i'm afraid i can't explain myself , sir . because i am not myself , you see ?")
		self.assertEqual(res[1], "i'm afraid i can't explain myself sir because i am not myself you see")
		self.assertEqual(res[2], "i'm_afraid afraid_i i_can't can't_explain explain_myself because_i i_am am_not not_myself you_see")
		self.assertEqual(res[3], "i'm_afraid_i afraid_i_can't i_can't_explain can't_explain_myself because_i_am i_am_not am_not_myself")

	def test_user_link(self):
		res = sanitize("/u/chiwong was here")
		self.assertEqual(res[0], "u/chiwong was here")
		self.assertEqual(res[1], "u/chiwong was here")
		self.assertEqual(res[2], "u/chiwong_was was_here")
		self.assertEqual(res[3], "u/chiwong_was_here")

	def test_link_in_parens(self):
		res = sanitize("this link (/u/omarTl) shouldn't be removed")
		self.assertEqual(res[0], "this link u/omartl shouldn't be removed")
		self.assertEqual(res[1], "this link u/omartl shouldn't be removed")
		self.assertEqual(res[2], "this_link link_u/omartl u/omartl_shouldn't shouldn't_be be_removed")
		self.assertEqual(res[3], "this_link_u/omartl link_u/omartl_shouldn't u/omartl_shouldn't_be shouldn't_be_removed")

	def test_user_link_remove(self):
		res = sanitize("hey check out this profile [chis profile](/u/chiwong)")
		self.assertEqual(res[0], "hey check out this profile chis profile")
		self.assertEqual(res[1], "hey check out this profile chis profile")
		self.assertEqual(res[2], "hey_check check_out out_this this_profile profile_chis chis_profile")
		self.assertEqual(res[3], "hey_check_out check_out_this out_this_profile this_profile_chis profile_chis_profile")
                
	def test_plain_url_with_www(self):
		res = sanitize("[omarTI](/u/omarTI)!!!!")
		self.assertEqual(res[0], "omarti!!!!")
		self.assertEqual(res[0], "omarti!!!!")
		self.assertEqual(res[2], "")
		self.assertEqual(res[3], "")

	def test_url_https_with_www(self):
		res = sanitize("this is a link to [reddit of the internet](https://www.reddit.com)")
		self.assertEqual(res[0], "this is a link to reddit of the internet")
		self.assertEqual(res[1], "this is a link to reddit of the internet")
		self.assertEqual(res[2], "this_is is_a a_link link_to to_reddit reddit_of of_the the_internet")
		self.assertEqual(res[3], "this_is_a is_a_link a_link_to link_to_reddit to_reddit_of reddit_of_the of_the_internet")

	def test_url_https_with_no_www(self):
		res = sanitize("this is a link to [reddit of the internet](https://reddit.com)")
		self.assertEqual(res[0], "this is a link to reddit of the internet")
		self.assertEqual(res[1], "this is a link to reddit of the internet")
		self.assertEqual(res[2], "this_is is_a a_link link_to to_reddit reddit_of of_the the_internet")
		self.assertEqual(res[3], "this_is_a is_a_link a_link_to link_to_reddit to_reddit_of reddit_of_the of_the_internet")
	
	def test_url_http_with_www(self):
		res = sanitize("this is a link to [reddit of the internet](http://www.reddit.com)")
		self.assertEqual(res[0], "this is a link to reddit of the internet")
		self.assertEqual(res[1], "this is a link to reddit of the internet")
		self.assertEqual(res[2], "this_is is_a a_link link_to to_reddit reddit_of of_the the_internet")
		self.assertEqual(res[3], "this_is_a is_a_link a_link_to link_to_reddit to_reddit_of reddit_of_the of_the_internet")

	def test_url_http_with_no_www(self):
		res = sanitize("this is a link to [reddit of the internet](http://reddit.com)")
		self.assertEqual(res[0], "this is a link to reddit of the internet")
		self.assertEqual(res[1], "this is a link to reddit of the internet")
		self.assertEqual(res[2], "this_is is_a a_link link_to to_reddit reddit_of of_the the_internet")
		self.assertEqual(res[3], "this_is_a is_a_link a_link_to link_to_reddit to_reddit_of reddit_of_the of_the_internet")
	
	def test_plain_url_with_www(self):
		res = sanitize("https://www.reddit.com")
		self.assertEqual(res[0], "")
		self.assertEqual(res[0], "")
		self.assertEqual(res[2], "")
		self.assertEqual(res[3], "")

	def test_plain_url_no_www(self):
		res = sanitize("https://reddit.com")
		self.assertEqual(res[0], "")
		self.assertEqual(res[0], "")
		self.assertEqual(res[2], "")
		self.assertEqual(res[3], "")

if __name__ == "__main__":
	# Just type 'python3 cleantext.py' to run unit tests
    if (len(sys.argv) > 1):
    	filename = sys.argv[1]
    	with open(filename) as f:
        	for i, line in enumerate(f):
	        	data_dict = json.loads(line)
	        	print(data_dict['body'])
	        	print(sanitize(data_dict['body']))
    else:
    	unittest.main()    	

