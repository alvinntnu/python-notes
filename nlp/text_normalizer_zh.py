import unicodedata
import re
#from nltk.corpus import wordnet
#import collections
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
import requests 
import pandas as pd

## Normalize unicode characters
def remove_weird_chars(text):
#     ```
#     (NFKD) will apply the compatibility decomposition, i.e. 
#     replace all compatibility characters with their equivalents. 
#     ```
    text = unicodedata.normalize('NFKD', text).encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    return text


## Remove duplicate spaces
def remove_extra_linebreaks(text):
    lines = text.split(r'\n+')
    return '\n'.join([re.sub(r'[\s]+',' ', l).strip() for l in lines if len(l)!=0])


def remove_extra_spaces(text):
    return re.sub("\\s+"," ", text).strip()

import jieba
jieba.set_dictionary('../../../RepositoryData/data/jiaba/dict.txt.jiebatw.txt')

## Word Segmentation
def seg(text, return_list = False):
    text_seg = jieba.cut(text)
    if return_list:
        out = [w for w in text_seg]
    else:
        out = ' '.join(text_seg)
    return out


def remove_symbols(text):
    text = re.sub('[\u0021-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e\u00a1-\u00bf\u2000-\u206f\u2013-\u204a\u20a0-\u20bf\u2100-\u214f\u2150-\u218b\u2190-\u21ff\u2200-\u22ff\u2300-\u23ff\u2460-\u24ff\u2500-\u257f\u2580-\u259f\u25a0-\u25ff\u2600-\u26ff\u2e00-\u2e7f\u3000-\u303f\ufe50-\ufe6f\ufe30-\ufe4f\ufe10-\ufe1f\uff00-\uffef─◆╱]+','',text)
    return text

def remove_numbers(text):
    return re.sub('\\d+',"", text)

def remove_alphabets(text):
    return re.sub('[a-zA-Z]+','', text)

def normalize_corpus(corpus, is_remove_extra_linebreaks=True,
                    is_remove_weird_chars=True,
                    is_seg=True,
                    is_remove_symbols=True,
                    is_remove_numbers=True,
                    is_remove_alphabets=True,
                    is_return_list = False):
    
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        
        if is_remove_extra_linebreaks:
            doc = remove_extra_linebreaks(doc)
            
        if is_remove_weird_chars:
            doc = remove_weird_chars(doc)
           
        if is_seg:
            doc=seg(doc, is_return_list)
            
        if is_remove_symbols:
            doc=remove_symbols(doc)
            
        if is_remove_alphabets:
            doc=remove_alphabets(doc)
            
        if is_remove_numbers:
            doc=remove_numbers(doc)
            
        normalized_corpus.append(remove_extra_spaces(doc))
        
    return normalized_corpus