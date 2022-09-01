#-*- encoding: utf-8 -*-
# generate all conjugation of a verb/auxiliary_verb/modal_verb
# must >> 'had to' is no conjugation
# added consonant doubling - jam
# unconjugatable word should be covered by irregular (must, ought)
# added functionality of finding infinitive when conjugated word in given
# check for the case when conjugated spell is the same as infinitive: fell/fall, found/find, lay/lie
# added show tenses with each conjugation (TS:third singular, PC:present continous, PA:past, PP:past participle)

import re
import os, sys
import pdb
from eng_inflection.get_comparative import count_syllable
from eng_inflection.define_conjugation import *
# reload(sys)
# sys.setdefaultencoding('utf-8')

def find_infinitive_irregular(word):
  for key in irregular_tagged_conjus:
    if word == key: return key
    conju_list = irregular_tagged_conjus[key]
    for conju in conju_list:
      if word == conju[3:]: return key
  return ""

def doubled_consonant(candidate):
  if count_syllable(candidate) == 1 and len(candidate) > 3 and candidate[-1] == candidate[-2]:
    word = candidate[:-1]
    if word[-1] in consonants and word[-1] != 'x' and word[-1] != 'w' and word[-2] in vowels and word[-3] in consonants:
      return True
  return False

# -ying should be covered by irregular conjugation
# -es, -ies, -ied should be covered by regular conjugation (kisses, fries, fried)
def find_infinitive_regular(word):
  if word.endswith('ing'):
    infinitive = word[:-3]
    if doubled_consonant(infinitive): infinitive = infinitive[:-1]
  elif word.endswith('ied'):
    infinitive = word[:-3] + 'y'
  elif word.endswith('ed'):
    infinitive = word[:-2]
    if doubled_consonant(infinitive): infinitive = infinitive[:-1]
  elif word.endswith('ies'):
    infinitive = word[:-3] + 'y'
  elif word.endswith('es'):
    infinitive = word[:-2]
  elif word.endswith('s'):
    infinitive = word[:-1]
  else:
    return ""
  conju_list = get_regular_conju(infinitive)
  for conju in conju_list:
    if word == conju[3:]: return infinitive
  return ""

def get_irregular_conju(word):
  if word in irregular_tagged_conjus:
    return irregular_tagged_conjus[word]
  else:
    return None

def most_regular(word):
  return ['TS:'+word+'s', 'PC:'+word+'ing', 'PA:'+word+'ed', 'PP:'+word+'ed']

# rule for doubling consonant
# two or more syllables should treated as irregular conjugation
def regular_one_syllable(word):
  if count_syllable(word) == 1 and len(word) > 2:
    if word[-1] in consonants and word[-1] != 'x' and word[-1] != 'w' and word[-2] in vowels and word[-3] in consonants:
      return ['TS:'+word+'s', 'PC:'+word+word[-1]+'ing', 'PA:'+word+word[-1]+'ed', 'PP:'+word+word[-1]+'ed']
  return None

def regular_endswith_e(word):
  if word[-1] == 'e':
    if word[-2] in ['e', 'y', 'o']: # free, dye, tiptoe
      return ['TS:'+word+'s', 'PC:'+word+'ing', 'PA:'+word+'d', 'PP:'+word+'d']
    elif word[-2] == 'i': # die, tie
      return ['TS:'+word+'s', 'PC:'+word[:-2]+'ying', 'PA:'+word+'d', 'PP:'+word+'d']
    else: # sue, love
      return ['TS:'+word+'s', 'PC:'+word[:-1]+'ing', 'PA:'+word+'d', 'PP:'+word+'d']
  else:
    return None

def regular_endswith_y(word):
  if word[-1] == 'y':
    if word[-2] in consonants:  # study, fly, carry, cry, worry, reply
      return ['TS:'+word[:-1]+'ies', 'PC:'+word+'ing', 'PA:'+word[:-1]+'ied', 'PP:'+word[:-1]+'ied']
    else: # enjoy, play, stay
      return most_regular(word)
  else:
    return None

# put es instead of s for 3rd person singular for a verb ends with s(ss), sh, ch, x, z, o
# kiss, buzz, box, polish, preach, solo
def regular_third_singular_es(word):
  if word.endswith('ch') or word.endswith('sh') or word[-1] == 'x' or word[-1] == 's' or word[-1] == 'z' or word[-1] == 'o':
    return ['TS:'+word+'es', 'PC:'+word+'ing', 'PA:'+word+'ed', 'PP:'+word+'ed']
  else:
    return None

# Verbs with more than one syllable ending in l double the final letter before suffixing in Britain English
# cancel > cancelled (US canceled)
# handled as irregular conjugation
def ends_l(word):
  pass

def regular_endswith_c(word):
  if word.endswith('c'): # panic, mimic, traffic, picnic
    return ['TS:'+word+'s', 'PC:'+word+'king', 'PA:'+word+'ked', 'PP:'+word+'ked']
  else:
    return None

def get_regular_conju(word):
  conju = regular_endswith_e(word)
  if conju == None: conju = regular_endswith_y(word)
  if conju == None: conju = regular_third_singular_es(word)
  if conju == None: conju = regular_endswith_c(word)
  if conju == None: conju = regular_one_syllable(word)
  if conju == None: conju = most_regular(word)
  return conju

def get_conjugation(word):
  if word in poly_conjus or word in irregular_tagged_conjus:
    infinitive = word
  else:
    infinitive = find_infinitive_irregular(word)
    if infinitive == "":
      if word.endswith('s'):
        infinitive = word
      else:
        infinitive = find_infinitive_regular(word)
      if infinitive == "":
        infinitive = word
  conju = get_irregular_conju(infinitive)
  if conju == None: conju = get_regular_conju(infinitive)
  return infinitive, conju

vowels = ['a','e', 'i', 'o', 'u']
consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'q', 'p', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']
poly_conjus = ['found', 'fell', 'lay', 'bind', 'saw', 'ground', 'wound', 'rent', 'resent']

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print ("need a verb to get conjugation")
    sys.exit()
  word = sys.argv[1]
  infinitive, conju = get_conjugation(word)
  print (infinitive, conju)
