#!/usr/bin/env python
# coding: utf-8

# # Natural Language Tool-Kits (NLTK)
# 
# - The almightly `nltk` package!

# ## Install
# 
# - Install package in terminal
# ```
# !pip install nltk
# ```
# - Download nltk data in python
# ```
# import nltk
# nltk.download('all', halt_on_error=False)
# ```

# In[4]:


import nltk
# nltk.download('all', halt_on_error=False)



# ## Corpora Data
# 
# - The package includes a lot of pre-loaded corpora datasets
# - The default `nltk_data` directory is in `/Users/YOUT_NAME/nltk_data/`
# - Selective Examples
#     - Brown Corpus
#     - Reuters Corpus
#     - WordNet 

# In[5]:


from nltk.corpus import gutenberg, brown, reuters

# brown corpus
## Categories (topics?)
print('Brown Corpus Total Categories: ', len(brown.categories()))
print('Categories List: ', brown.categories())


# In[6]:


# Sentences
print(brown.sents()[0]) ## first sentence
print(brown.sents(categories='fiction')) ## first sentence for fiction texts


# In[7]:


## Tagged Sentences
print(brown.tagged_sents()[0])


# In[8]:


## Sentence in natural forms
sents = brown.sents(categories='fiction')
[' '.join(sent) for sent in sents[1:5]]


# In[9]:


## Get tagged words
tagged_words = brown.tagged_words(categories='fiction')

#print(tagged_words[1]) ## a tuple

## Get all nouns 
nouns = [(word, tag) for word, tag in tagged_words 
                      if any (noun_tag in tag for noun_tag in ['NP','NN'])]
## Check first ten nouns
nouns[:10]


# In[10]:


## Creating Freq list
nouns_freq = nltk.FreqDist([w for w, t in nouns])
sorted(nouns_freq.items(),key=lambda x:x[1], reverse=True)[:20]


# In[11]:


sorted(nouns_freq.items(),key=lambda x:x[0], reverse=True)[:20]


# In[12]:


nouns_freq.most_common(10)


# In[13]:


## Accsess data via fileid
brown.fileids(categories='fiction')[0]
brown.sents(fileids='ck01')


# ## WordNet
# 
# - A dictionary resource

# In[14]:


from nltk.corpus import wordnet as wn
word = 'walk'

# get synsets
word_synsets = wn.synsets(word)
word_synsets


# In[15]:


## Get details of each synset
for s in word_synsets:
    if str(s.name()).startswith('walk.v'):
        print(
            'Syset ID: %s \n'
            'POS Tag: %s \n'
            'Definition: %s \n'
            'Examples: %s \n' % (s.name(), s.pos(), s.definition(),s.examples())
        )

