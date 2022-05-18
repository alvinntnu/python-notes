#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[7]:


from datetime import date
print("Last updated: " + date.today().strftime("%B %d, %Y"))


# ```{admonition} My Personal Python Notes!
# :class: tip
# 
# This notebook collects my personal Python notes for linguistics. These notes are based on many different sources. Please note that these notes may have not been organized in a very systematic way. This notebook is being updated on a regular basis. Most of the materials here are for educational purposes only.
# 
# ```

# ## Table of Contents
# 
# There are four main sections in Python Notes for Linguistics:
# 
# ````{panels}
# <i class="fa fa-check fa-1x" style="color:DarkTurquoise;margin-right:5px"></i>[Python Basics](python-basics/python-basics)
# ^^^
# This section covers the fundamental concepts of the Python language.
# 
# ---
# 
# <i class="fa fa-check fa-1x" style="color:DarkTurquoise;margin-right:5px"></i>[Corpus Linguistics with Python](corpus/corpus-processing)
# ^^^
# This section covers the corpus processing skills and techniques using Python.
# 
# ---
# 
# <i class="fa fa-check fa-1x" style="color:DarkTurquoise;margin-right:5px"></i> [Statistical Analyses](statistical-analyses/statistical-analyses)
# ^^^
# This section covers statistical analyses using Python.
# 
# --- 
# 
# <i class="fa fa-check fa-1x" style="color:DarkTurquoise;margin-right:5px"></i> [NLP with Python](nlp/nlp)
# ^^^
# This section covers computational text analytics using Python.
# 
# ````

# ## Useful Resources
# 
# There are many resources for Python-learning. Here I keep track of a few useful sources of inspirations for my Python journey.

# ```{tabbed} Books
# 
# The materials collected here were based on several useful reference books, which are listed as follows in terms of three categories:
# 
# - Python Basics {cite}`gerrard2016lean`
# - Data Analysis with Python {cite}`mckinney2012python`
# - Natural Language Processing with Python {cite}`sarkar2019text,bird2009natural,vajjala2020,perkins2014python,srinivasa2018natural`
# - Deep Learning {cite}`francois2017deep`
# ```
# 
# 
# ```{tabbed} Online Resources
# 
# In addition to books, there are many wonderful on-line resources, esp. professional blogs, providing useful tutorials and intuitive understanding of many complex ideas in NLP and AI development. Among them, here is a list of my favorites:
# 
# - [Toward Data Science](https://towardsdatascience.com/)
# - [LeeMeng](https://leemeng.tw/)
# - [Dipanzan Sarkar's articles](https://towardsdatascience.com/@dipanzan.sarkar)
# - [Python Graph Libraries](https://python-graph-gallery.com/)
# - [KGPTalkie NLP](https://kgptalkie.com/category/natural-language-processing-nlp/)
# - [Jason Brownlee's Blog: Machine Learning Mastery](https://machinelearningmastery.com/)
# - [Jay Alammar's Blog](https://jalammar.github.io/)
# - [Chris McCormich's Blog](https://mccormickml.com/)
# - [GLUE: General Language Understanding Evaluation Benchmark](https://gluebenchmark.com/)
# - [SuperGLUE](https://super.gluebenchmark.com/)
# 
# ```
# 
# 
# 
# ```{tabbed} YouTube Channels
# 
# - [Chris McCormick AI](https://www.youtube.com/channel/UCoRX98PLOsaN8PtekB9kWrw/videos)
# - [Corey Schafer](https://www.youtube.com/channel/UCCezIgC97PvUuR4_gbFUs5g)
# - [Edureka!](https://www.youtube.com/channel/UCkw4JCwteGrDHIsyIIKo4tQ)
# - [Deeplearning.ai](https://www.youtube.com/channel/UCcIXc5mJsHVYTZR1maL5l9w)
# - [Free Code Camp](https://www.youtube.com/c/Freecodecamp/videos)
# - [Giant Neural Network](https://www.youtube.com/watch?v=ZzWaow1Rvho&list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So)
# - [Tech with Tim](https://www.youtube.com/channel/UC4JX40jDee_tINbkjycV4Sg)
# - [PyData](https://www.youtube.com/user/PyDataTV/featured)
# - [Python Programmer](https://www.youtube.com/channel/UC68KSmHePPePCjW4v57VPQg)
# - [Keith Galli](https://www.youtube.com/channel/UCq6XkhO5SZ66N04IcPbqNcw)
# - [Data Science Dojo](https://www.youtube.com/c/Datasciencedojo/featured)
# - [Calculus: Single Variable by Professor Robert Ghrist](https://www.youtube.com/playlist?list=PLKc2XOQp0dMwj9zAXD5LlWpriIXIrGaNb)
# 
# ```

# ## References
# 
# ```{bibliography} book.bib
# :filter: docname in docnames
# :style: unsrt
# ```
