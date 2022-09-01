#-*- encoding: utf-8 -*-
# uncountable nouns is treated as irregular - news (cefr parser make uncountable irregular)
# uncountable should listed separately
# added functionality of finding singular when input is plural
# may not work correctly when singular ends with e > lets give two possible singulars
# default plural for nouns ends with o is putting s not putting es
import os, sys
import pdb
from eng_inflection.define_plural import *
# reload(sys)
# sys.setdefaultencoding('utf-8')

def find_singular_irregular(word):
  for key in irregular_tagged_plurals:
    if word == key: return key
    plural_list = irregular_tagged_plurals[key]
    for plural in plural_list:
      if word == plural[3:]: return key
  return ""

# cannot differenciate se+s/s+es, che+s/ch+es, she+s/sh+es, xe+s/x+es, ze+s/z+es, oe+s/o+es
def find_singular_regular(word):
  if word.endswith('ies'):
    singular = word[:-3] + 'y'
  elif word.endswith('es') and (word[:-2].endswith('ch') or word[:-2].endswith('sh') or word[-3] == 'x' \
    or word[-3] == 's' or word[-3] == 'z' or word[-3] == 'o'):
    singular = word[:-2]
  elif word.endswith('s'):
    singular = word[:-1]
  else:
    return ""
  plural_list = get_regular_plural(singular)
  for plural in plural_list:
    if word == plural[3:]:
      return singular
  return ""

def get_irregular_plural(word):
  if word in irregular_tagged_plurals:
    return irregular_tagged_plurals[word]
  else:
    return None

# ~man, ~in-law
# what others?
def get_compound_plural(word):
  if (word.endswith("man") and len(word) > 5) or (word.endswith("woman") and len(word) > 7):
    return ['PL:'+word[:-3]+'men']
  if (word.endswith("-in-law")):
    if word[:-7] in irregular_tagged_plurals:
      plural_array = irregular_tagged_plurals[word[:-7]]
    else:
      plural_array = get_regular_plural(word[:-7])
    for entry in plural_array:
      plural.append(entry + "-in-law")
    return plural
  return None

def regular_plural(word):
  return ['PL:'+word+'s']

def ends_y(word):
  if word[-1] == 'y':
    if word[-2] in consonants:
      return ['PL:'+word[:-1]+'ies']
    else:
      return regular_plural(word)
  else:
    return None

def ends_s(word):
  if word.endswith('ch') or word.endswith('sh') or word[-1] == 'x' or word[-1] == 's' or word[-1] == 'z':
    return ['PL:'+word+'es']
  else:
    return None

# default is putting just 's' not 'es'
def ends_o(word):
  if word.endswith('o'):
    return ['PL:'+word+'s']
  else:
    return None

def get_regular_plural(word):
  plural = ends_s(word)
  if plural == None: plural = ends_y(word)
  if plural == None: plural = ends_o(word)
  if plural == None: plural = regular_plural(word)
  return plural

def get_plural(word):
  singular = find_singular_irregular(word)
  if singular == "":
    singular = find_singular_regular(word)
    if singular == "":
      singular = word

  plural = get_irregular_plural(singular)
  if plural == None: plural = get_compound_plural(singular)
  if plural == None: plural = get_regular_plural(singular)
  if len(plural) == 1 and plural[0][3:] == singular:
    plural = []
  return singular, plural

vowels = ['a','e', 'i', 'o', 'u']
consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'q', 'p', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']
es_exceptions = ['expense']

if __name__ == "__main__":
  # if len(sys.argv) < 2:
  #   print ("need a noun to get plural")
  #   sys.exit()
  word = "knife"
  singular, plural = get_plural(word)
  print (singular, plural)
