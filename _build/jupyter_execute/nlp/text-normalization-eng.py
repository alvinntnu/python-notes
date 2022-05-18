#!/usr/bin/env python
# coding: utf-8

# # Text Normalization (English)
# 

# In[1]:


import spacy
import unicodedata
#from contractions import CONTRACTION_MAP
import re
from nltk.corpus import wordnet
import collections
#from textblob import Word
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup


# ## HTML Tags

# In[12]:


import requests
from bs4 import BeautifulSoup

data = requests.get('https://en.wikipedia.org/wiki/Python_(programming_language)')
content = data.content
print(content[:500])


# In[11]:


def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    ## Can include more HTML preprocessing here...
    stripped_html_elements = soup.findAll(name='div',attrs={'id':'mw-content-text'})
    stripped_text = ' '.join([h.get_text() for h in stripped_html_elements])
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

clean_content = strip_html_tags(content)
print(clean_content[:500])


# ## Stemming

# ## Lemmatization
# 

# In[4]:


import spacy
# use spacy.load('en') if you have downloaded the language model en directly after install spacy
nlp = spacy.load('en', parse=False, tag=True, entity=False)
text = 'My system keeps crashing his crashed yesterday, ours crashes daily'

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

lemmatize_text("My system keeps crashing! his crashed yesterday, ours crashes daily")


# ## Contractions

# In[5]:


from contractions import CONTRACTION_MAP
import re

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


# In[6]:


expand_contractions("Y'all can't expand contractions I'd think")

expand_contractions("I'm very glad he's here!")


# ## Accented Characters (Non-ASCII)
# 
# - [unicodedata dcoumentation](https://docs.python.org/3/library/unicodedata.html)

# In[7]:


import unicodedata

def remove_accented_chars(text):
#     ```
#     (NFKD) will apply the compatibility decomposition, i.e. 
#     replace all compatibility characters with their equivalents. 
#     ```
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


remove_accented_chars('Sómě Áccěntěd těxt')

# print(unicodedata.normalize('NFKD', 'Sómě Áccěntěd těxt'))
# print(unicodedata.normalize('NFKD', 'Sómě Áccěntěd těxt').encode('ascii','ignore'))
# print(unicodedata.normalize('NFKD', 'Sómě Áccěntěd těxt').encode('ascii','ignore').decode('utf-8', 'ignore'))


# ## Special Characters

# ## Stopwords

# In[8]:


import nltk
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer() # assuming per line per sentence 
    # for other Tokenizer, maybe sent.tokenize should go first
stopword_list = nltk.corpus.stopwords.words('english')
def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

remove_stopwords("The, and, if are stopwords, computer is not")


# ## Redundant Whitespaces

# ## Spelling Checks
