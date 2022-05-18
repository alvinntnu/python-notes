#!/usr/bin/env python
# coding: utf-8

# # Phythonic Productivity Techniques

# ## Getting Help

# In[1]:


import datetime
dir(datetime)
dir


# In[3]:


# Check objects with specific keywords
[_ for _ in dir(datetime) if 'date' in _.lower()]


# ## Virtual Environments

# The following codes are run under terminal.
# 
# The following codes show how to create a virtual environment in the terminal.
# 
# ```
# which pip
# 
# pytho3 -m venv `DIRECTORY`
# ls `DIRECTORY`
# 
# source activate `DIRECTORY`
# pip list
# pip install `PACKAGE_NAME`
# source deactivate
# 
# ```
