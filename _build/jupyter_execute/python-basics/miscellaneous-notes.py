#!/usr/bin/env python
# coding: utf-8

# # Miscellaneous Notes

# - Fundamentals
# - Data Structure
# - Program Structure
# - Input and Output
# - Classes and Objects

# In[11]:


## How to check current object sizes in Notebook
import sys
a = range(10000)
print(sys.getsizeof(a))


# In[14]:


## Get the docstring
get_ipython().run_line_magic('pinfo', 'sys.getsizeof')
get_ipython().run_line_magic('pinfo', 'who')


# In[ ]:


get_ipython().run_line_magic('who', '')
get_ipython().run_line_magic('whos', '')


# In[6]:


globals()


# In[8]:


locals()


# In[9]:


dir()

