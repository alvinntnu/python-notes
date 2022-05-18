#!/usr/bin/env python
# coding: utf-8

# # Praat TextGrid Data
# 
# - Module: [praatio](https://pypi.org/project/praatio/)
# - Tutorial: [PraatIO-Doing Speech Analysis with Python](https://nbviewer.jupyter.org/github/timmahrt/praatIO/blob/master/tutorials/tutorial1_intro_to_praatio.ipynb)

# In[3]:


from praatio import tgio
tg = tgio.openTextgrid('../../../../../Dropbox/Projects/MOST-Prosody/data/2014_di702_TextGrid_Alvin/di_001.TextGrid')


# In[4]:


tg.tierDict


# In[5]:


# get all intervals
tg.tierDict['PU'].entryList


# In[6]:


tg.tierDict['Word'].entryList


# In[7]:


tg.tierDict['Word'].find('我')


# In[8]:


import pandas as pd
word_tier = tg.tierDict['Word']
pd.DataFrame([(start, end, label) for (start, end, label) in word_tier.entryList],
            columns = ['start','end','label'])


# In[9]:


pu_tier = tg.tierDict['PU']
pd.DataFrame([(start, end, label) for (start, end, label) in pu_tier.entryList],
            columns = ['start','end','label'])

