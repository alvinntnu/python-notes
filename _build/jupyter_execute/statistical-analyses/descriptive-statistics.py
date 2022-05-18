#!/usr/bin/env python
# coding: utf-8

# # Descriptive Statistics

# In[6]:


DEMO_DATA_ROOT = "../../../RepositoryData/data"


# In[3]:


import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ## Univariate Statistics

# In[11]:


UHM = pd.read_table(DEMO_DATA_ROOT+"/gries_sflwr/_inputfiles/03-1_uh(m).csv")
UHM


# In[15]:


UHM.value_counts(UHM['FILLER'])


# In[16]:


UHM.value_counts(UHM['FILLER'], normalize=True)


# In[21]:


def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x)+1)/len(x) # percentiles
    return(x,y)

ecdf(UHM.value_counts(UHM['FILLER']))

