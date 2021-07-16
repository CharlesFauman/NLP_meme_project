
# coding: utf-8

# In[1]:

import glob
import os
import re

import pickle

import csv
import pandas as pd
import numpy as np

from unidecode import unidecode

import string

import language_check

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

import cleaning_functions


# In[2]:

meta_dict = dict()
meta_dict["language_check_tool"] = language_check.LanguageTool('en-US')
meta_dict["lemmatizer"] = WordNetLemmatizer()
meta_dict["stop_words_en"] = set(stopwords.words('english'))

column_names = ['type','len', 'word', 'POS', 'stemmed', 'priorpolarity']
sentiments = pd.read_csv("sentiment/sentiment.txt", sep = " ", header=None, names = column_names)
sentiments['type'] = [x[5:] for x in sentiments['type']]
sentiments['len'] = [x[4:] for x in sentiments['len']]
sentiments['word'] = [x[6:] for x in sentiments['word']]
sentiments['POS'] = [x[5:] for x in sentiments['POS']]
sentiments['stemmed'] = [x[9:] for x in sentiments['stemmed']]
sentiments['priorpolarity'] = [x[14:] for x in sentiments['priorpolarity']]

meta_dict["sentiments"] = sentiments

pickle.dump(meta_dict, open( "meta_dict.p", "wb" ) )


# In[4]:

meme_annotations = dict()
for filename in glob.glob('../collection/annotations 2/*.txt'):
    with open(filename, 'r', encoding="utf8") as annotations_file:
        annotations_list = annotations_file.read().lower().splitlines()
        
        data = {'id' : [meme[:7] for meme in annotations_list],
                'text' : [meme[9:-1] for meme in annotations_list]}
        meme_annotations[os.path.basename(filename)[:-4]] = pd.DataFrame(data)
cleaned_data = dict()


# In[5]:

for meme_name, meme_df in meme_annotations.items():
    print(meme_name + ":")
    meme_cleaned_data = list()
    for index, row in meme_df.iterrows():
        if(index % 50 == 0): print(index)
        lemmas = cleaning_functions.clean_sentence(meta_dict, row['text'][1:])
        if(len(lemmas) == 0):
            continue
        meme_cleaned_data.append(lemmas)
    cleaned_data[meme_name] = meme_cleaned_data


# In[7]:

with open('cleaned_memes.tsv', 'w', encoding="utf-8") as output:
    
    output.write("meme")
    output.write('\t')
    output.write("meme_id")
    output.write('\t')
    output.write("token")
    output.write('\t')
    output.write("pos")
    output.write('\t')
    output.write("sentiment")
    output.write('\n')
    
    for meme_name, cleaned_memes in cleaned_data.items():
        for counter, meme in enumerate(cleaned_memes):
            for token in meme:
                output.write(meme_name)
                output.write('\t')
                output.write(str(counter))
                output.write('\t')
                output.write(token[0])
                output.write('\t')
                output.write(token[1])
                output.write('\t')
                output.write(token[2])
                output.write('\n')

