#!/usr/bin/env python
# coding: utf-8

# # WordNet

# WordNet is a lexical database for the English language, where word senses are connected as a systematic lexical network.

# ## Import

# In[1]:


from nltk.corpus import wordnet


# ## Synsets

# A `synset` has several attributes, which can be extracted via its defined methods:
# 
# - `synset.name()`
# - `synset.definition()`
# - `synset.hypernyms()`
# - `synset.hyponyms()`
# - `synset.hypernym_path()`
# - `synset.pos()`

# In[6]:


syn = wordnet.synsets('walk', pos='v')[0]
print(syn.name())
print(syn.definition())


# In[7]:


syn.examples()


# In[8]:


syn.hypernyms()


# In[10]:


syn.hypernyms()[0].hyponyms()


# In[12]:


syn.hypernym_paths()


# In[14]:


syn.pos()


# ## Lemmas 

# A `synset` may coreespond to more than one lemma.

# In[19]:


syn = wordnet.synsets('walk', pos='n')[0]
print(syn.lemmas())


# Check the lemma names.

# In[23]:


for l in syn.lemmas():
    print(l.name())


# ## Synonyms

# In[29]:


synonyms = []
for s in wordnet.synsets('run', pos='v'):
    for l in s.lemmas():
        synonyms.append(l.name())
print(len(synonyms))
print(len(set(synonyms)))

print(set(synonyms))


# ## Antonyms

# Some lemmas have antonyms.
# 
# The following examples show how to find the antonyms of `good` for its two different senses, `good.n.02` and `good.a.01`.

# In[37]:


syn1 = wordnet.synset('good.n.02')
syn1.definition()


# In[38]:


ant1 = syn1.lemmas()[0].antonyms()[0]


# In[39]:


ant1.synset().definition()


# In[54]:


ant1.synset().examples()


# In[48]:


syn2 = wordnet.synset('good.a.01')
syn2.definition()


# In[50]:


ant2 = syn2.lemmas()[0].antonyms()[0]


# In[51]:


ant2.synset().definition()


# In[53]:


ant2.synset().examples()


# ## Wordnet Synset Similarity

# With a semantic network, we can also compute the semantic similarty between two synsets based on their distance on the tree. 
# 
# In particular, this is possible cause all synsets are organized in a hypernym tree.
# 
# The recommended distance metric is Wu-Palmer Similarity (i.e., `synset.wup_similarity()`)

# In[62]:


s1 = wordnet.synset('walk.v.01')
s2 = wordnet.synset('run.v.01')
s3 = wordnet.synset('toddle.v.01')


# In[63]:


s1.wup_similarity(s2)


# In[64]:


s1.wup_similarity(s3)


# In[65]:


s1.common_hypernyms(s3)


# In[66]:


s1.common_hypernyms(s2)


# Two more metrics for lexical semilarity:
# 
# - `synset.path_similarity()`: Path Similarity
# - `synset.lch_similarity()`: Leacock Chordorow Similarity
