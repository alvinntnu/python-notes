# Text Normalization (Chinese)

- `text_normalizer_zh.py`
- Including functions for:
    - word-seg chinese texts
    - clean up texts by removing duplicate spaces and line breaks
    - remove incompatible weird characters

import unicodedata
import re
#from nltk.corpus import wordnet
#import collections
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
import requests
import pandas as pd
import text_normalizer_zh as tnz

## Normalization Functions

# %load text_normalizer_zh.py
"""

Notes
-----

    These functions are based on the text normalization functions 
    provided in Text Analytics with Python 2ed.

"""

import unicodedata
import re
# from nltk.tokenize.toktok import ToktokTokenizer
import pandas as pd
import jieba

## Initialize Trad Chinese dictionary
jieba.set_dictionary('../../../RepositoryData/data/jiaba/dict.txt.jiebatw.txt')


## Normalize unicode characters
def remove_weird_chars(text):
    #     ```
    #     (NFKD) will apply the compatibility decomposition, i.e.
    #     replace all compatibility characters with their equivalents.
    #     ```
    text = unicodedata.normalize('NFKD', text).encode('utf-8',
                                                      'ignore').decode(
                                                          'utf-8', 'ignore')
    return text


## Remove extra linebreaks
def remove_extra_linebreaks(text):
    lines = text.split(r'\n+')
    return '\n'.join(
        [re.sub(r'[\s]+', ' ', l).strip() for l in lines if len(l) != 0])


## Remove extra medial/trailing/leading spaces
def remove_extra_spaces(text):
    return re.sub("\\s+", " ", text).strip()


## Seg the text into words
def seg(text):
    text_seg = jieba.cut(text)
    out = ' '.join(text_seg)
    return out


## Remove punctuation/symbols
def remove_symbols(text):
    """
    
    Unicode 6.0 has 7 character categories, and each category has subcategories:

    Letter (L): lowercase (Ll), modifier (Lm), titlecase (Lt), uppercase (Lu), other (Lo)
    Mark (M): spacing combining (Mc), enclosing (Me), non-spacing (Mn)
    Number (N): decimal digit (Nd), letter (Nl), other (No)
    Punctuation (P): connector (Pc), dash (Pd), initial quote (Pi), final quote (Pf), open (Ps), close (Pe), other (Po)
    Symbol (S): currency (Sc), modifier (Sk), math (Sm), other (So)
    Separator (Z): line (Zl), paragraph (Zp), space (Zs)
    Other (C): control (Cc), format (Cf), not assigned (Cn), private use (Co), surrogate (Cs)
    
    
    There are 3 ranges reserved for private use (Co subcategory): 
    U+E000—U+F8FF (6,400 code points), U+F0000—U+FFFFD (65,534) and U+100000—U+10FFFD (65,534). 
    Surrogates (Cs subcategory) use the range U+D800—U+DFFF (2,048 code points).
    
    
    """

    ## Brute-force version: list all possible unicode ranges, but this list is not complete.
    #   text = re.sub('[\u0021-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e\u00a1-\u00bf\u2000-\u206f\u2013-\u204a\u20a0-\u20bf\u2100-\u214f\u2150-\u218b\u2190-\u21ff\u2200-\u22ff\u2300-\u23ff\u2460-\u24ff\u2500-\u257f\u2580-\u259f\u25a0-\u25ff\u2600-\u26ff\u2e00-\u2e7f\u3000-\u303f\ufe50-\ufe6f\ufe30-\ufe4f\ufe10-\ufe1f\uff00-\uffef─◆╱]+','',text)

    text = ''.join(ch for ch in text
                   if unicodedata.category(ch)[0] not in ['P', 'S'])
    return text


## Remove numbers
def remove_numbers(text):
    return re.sub('\\d+', "", text)


## Remove alphabets
def remove_alphabets(text):
    return re.sub('[a-zA-Z]+', '', text)


## Combine every step
def normalize_corpus(corpus,
                     is_remove_extra_linebreaks=True,
                     is_remove_weird_chars=True,
                     is_seg=True,
                     is_remove_symbols=True,
                     is_remove_numbers=True,
                     is_remove_alphabets=True):

    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:

        if is_remove_extra_linebreaks:
            doc = remove_extra_linebreaks(doc)

        if is_remove_weird_chars:
            doc = remove_weird_chars(doc)

        if is_seg:
            doc = seg(doc)

        if is_remove_symbols:
            doc = remove_symbols(doc)

        if is_remove_alphabets:
            doc = remove_alphabets(doc)

        if is_remove_numbers:
            doc = remove_numbers(doc)

        normalized_corpus.append(remove_extra_spaces(doc))

    return normalized_corpus

## Extract an Article

- Grab the first article from Google news for demonstration

url = 'https://news.google.com/topics/CAAqJQgKIh9DQkFTRVFvSUwyMHZNRFptTXpJU0JYcG9MVlJYS0FBUAE?hl=zh-TW&gl=TW&ceid=TW%3Azh-Hant'
r = requests.get(url)
web_content = r.text
soup = BeautifulSoup(web_content, 'lxml')
title = soup.find_all('a', class_='DY5T1d')
first_art_link = title[0]['href'].replace('.', 'https://news.google.com', 1)

#print(first_art_link)
art_request = requests.get(first_art_link)
art_request.encoding = 'utf8'
soup_art = BeautifulSoup(art_request.text, 'lxml')

art_content = soup_art.find_all('p')
art_texts = [p.text for p in art_content]
print(art_texts)

- Normalized results:

tnz.normalize_corpus([' '.join(art_texts)])

