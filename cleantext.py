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
    text = re.sub('\s+',' ',text)
    # Remove URLs
    text = re.sub('(http[s]?://)?www.\S+', '', text)    
    # Remove links to subreddits and users
    text = re.sub('\/r\/[_\-a-z0-9A-Z]*', '', text)
    text = re.sub('\/u\/[_\-a-z0-9A-Z]*', '', text)
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
    plain_tokens = []
    for index, token in enumerate(tokens):
    	parsed_text += token + ' '
    	if token not in common:
    		unigrams += token + ' '
    		if index + 1 <= len(tokens)-1 and tokens[index+1] not in common:
    			bigrams += token + '_' + tokens[index+1] + ' '
    			if index + 2 <= len(tokens)-1 and tokens[index+2] not in common:
    				trigrams += token + '_' + tokens[index+1] + '_' + tokens[index+2] + ' '
    
    # print('Parsed Text:\n'+ parsed_text + '\n')
    # print('Unigrams:\n'+ unigrams + '\n')
    # print('Bigrams:\n'+ bigrams +'\n')
    # print('Trigrams:\n'+ trigrams + '\n')

    # Separate all external punctuation such as periods, commas, etc. into their own tokens (a token is a single piece of text with no spaces), but maintain punctuation within words
    return [parsed_text.strip(), unigrams.strip(), bigrams.strip(), trigrams.strip()]


def separate_tokens(word, token_list):
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
		res = sanitize("wow\n this looks really \n cool \njoinme?")
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
		res = sanitize("I'm afraid I can't explain myself, sir. Because I am not myself, you see?")
		self.assertEqual(res[0], "i'm afraid i can't explain myself , sir . because i am not myself , you see ?")
		self.assertEqual(res[1], "i'm afraid i can't explain myself sir because i am not myself you see")
		self.assertEqual(res[2], "i'm_afraid afraid_i i_can't can't_explain explain_myself because_i i_am am_not not_myself you_see")
		self.assertEqual(res[3], "i'm_afraid_i afraid_i_can't i_can't_explain can't_explain_myself because_i_am i_am_not am_not_myself")

	'''
	Failing test:
		Check the regex code, it seemed like /u/chiwong gets removed completely 
	'''
	def test_user_link(self):
		res = sanitize("/u/chiwong was here")
		self.assertEqual(res[0], "u/chiwong was here")
		self.assertEqual(res[1], "u/chiwong was here")
		self.assertEqual(res[2], "u/chiwong_was was_here")
		self.assertEqual(res[3], "u/chiwong_was_here")

	'''
	Failing test:
		This is how it should output like
		[desc of url](url) --> desc of url
	'''
	def test_url(self):
		right_link_answer = "congress specifically passed a law removing all consumers right to sue"
		res = sanitize("""[Congress specifically passed a law removing all consumers right to sue]
			(https://techcrunch.com/2017/10/24/congress-votes-to-disallow-consumers-from-suing-equifax-and-other-companies-with-arbitration-agreements)""")
		self.assertEqual(res[0], right_link_answer)
		res = sanitize(right_link_answer)
		# Run tests on what the sanitized link should have been, these will and should pass
		self.assertEqual(res[1], "congress specifically passed a law removing all consumers right to sue")
		self.assertEqual(res[2], "congress_specifically specifically_passed passed_a a_law law_removing removing_all all_consumers consumers_right right_to to_sue")
		self.assertEqual(res[3], "congress_specifically_passed specifically_passed_a passed_a_law a_law_removing law_removing_all removing_all_consumers all_consumers_right consumers_right_to right_to_sue")


if __name__ == "__main__":
    # This is the Python main function.
    # You should be able to run
    # python cleantext.py <filename>
    # and this "main" function will open the file,
    # read it line by line, extract the proper value from the JSON,
    # pass to "sanitize" and print the result as a list.
    
    # # YOUR CODE GOES BELOW.
    # sanitize("what !!!!! they am i where")
    if (len(sys.argv) > 1):
    	print("""Chi, read this.
    			\nCheck out the unit tests in here and type python3 cleantext.py to run them
    			\n2 of them are currently failing (url and user/sub links)""")
    	exit()
    unittest.main()

    # sanitize("")
    # sanitize("what is the meaning of life?")
    # parser = argparse.ArgumentParser()
    # parser.add_argument('filename')
    # filename = parser.parse_args().filename
    # # Extract comment from file line by line
    # with open(filename) as f:
    #     for i, line in enumerate(f):
    #             line = f.readline()
    #             data_dict = json.loads(line)
    #             print(data_dict['body'])
    #             print(sanitize(data_dict['body']))

