#!/usr/bin/env python
# coding: utf-8

# # Object Serialization
# 
# - `pickle` python objects for later use

# In[1]:


import pickle

example_dict = {1:"6",2:"2",3:"f"}

pickle_out = open("dict.pickle","wb")
pickle.dump(example_dict, pickle_out)
pickle_out.close()


# In[2]:


pickle_in = open("dict.pickle","rb")
example_dict = pickle.load(pickle_in)


# In[3]:


get_ipython().system('rm dict.pickle')

