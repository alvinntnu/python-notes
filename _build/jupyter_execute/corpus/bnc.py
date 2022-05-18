#!/usr/bin/env python
# coding: utf-8

# # BNC-XML
# 
# - XML
# - CHILDES
# - JSON

# ## BNC XML

# In[11]:


## Read BNC XML
import nltk
from nltk.corpus.reader.bnc import BNCCorpusReader
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

# Instantiate the reader like this
bnc_reader = BNCCorpusReader(root="../../../Corpus/BNC-XML/Texts/", fileids=r'[A-K]/\w*/\w*\.xml')
list_of_fileids = ['A/A0/A00.xml', 'A/A0/A01.xml']
bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(bnc_reader.words(fileids=list_of_fileids))
scored = finder.score_ngrams(bigram_measures.raw_freq)

print(scored)

