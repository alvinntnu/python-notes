#!/usr/bin/env python
# coding: utf-8

# # Regular Expression

# - This notebook introduces the powerful regular expressions for pattern matching.
# - Codes credit to [Corey Schaffer's tutorial on regular expressions](https://www.youtube.com/watch?v=sa-TUpSx1JA)

# ## Comparison of Python and R

# |Python   |R     |
# |---|---|
# |`re.search()`|`str_extract()`|
# |`re.findall()`|`str_extract_all()`|
# |`re.finditer()`|`str_match_all()`|
# |`re.sub()`|`str_replace_all()`|
# |`re.split()`|`str_split()`|
# |`re.subn()`|?|
# |`re.match()`|?|
# |?|`str_detect()`|
# |?|`str_subset()`|

# The above table shows the similarities and differences in terms of the regular expression functions in Python and R. They are more or less similar. These mappings can be helpful for R users to understand the `re` in Python.

# ## Regular Expression Syntax

# In[1]:


import re

text_to_search = '''
abcdefghijklmnopqurtuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
1234567890
Ha HaHa
MetaCharacters (Need to be escaped):
. ^ $ * + ? { } [ ] \ | ( )
coreyms.com
321-555-4321
123.555.1234
123*555*1234
800-555-1234
900-555-1234
Mr. Schafer
Mr Smith
Ms Davis
Mrs. Robinson
Mr. T
'''

sentence = 'Start a sentence and then bring it to an end'

pattern = re.compile(r'\d{3}-\d{3}-\d{4}', re.I)


# In[46]:


## perform a search
matches= re.search(pattern, text_to_search)
if matches:
    print(matches.group())


# In[51]:


## find all matches
matches = re.findall(pattern, text_to_search)
if matches:
    for m in matches:
        print(m.strip())


# In[57]:


## find all matches
matches = re.finditer(pattern, text_to_search)
if matches:
    for m in matches:
        print("%02d-%02d: %s" % (m.start(), m.end(), m.group()))


# ## Regular Expression in Python

# ### Raw String Notation
# 
# Raw string notation (r"text") keeps regular expressions sane. Without it, every backslash ('\') in a regular expression would have to be prefixed with another one to escape it. For example, the two following lines of code are functionally identical:

# ### Find all matches

# - `re.findall()`: matches all occurrences of a pattern, not just the first one as search() does.
# - `re.finditer(): If one wants more information about all matches of a pattern than the matched text, finditer() is useful as it provides match objects instead of strings.

# ### group() vs. groups()
# 
# - `group()`: by default, returns the match
# - `groups()`: by default, returns all capturing groups

# In[36]:


m = re.match("a(.)(.)","abcedf")

print(m.group(0)) # return the whole match
print(m.group()) # return the whole match, same as above
print(m.groups()) # return each capturing group match
print(m.group(1)) # return first capturing gorup match


# ### string format validation

# In[41]:


valid = re.compile(r"^[a-z]+@[a-z]+\.[a-z]{3}$")
print(valid.match('alvin@ntnu.edu'))
print(valid.match('alvin123@ntnu.edu'))
print(valid.match('alvin@ntnu.homeschool'))


# ### re.match() vs. re.search()

# Python offers two different primitive operations based on regular expressions: 
# - `re.match()` checks for a match only at the beginning of the string
# - `re.search()` checks for a match anywhere in the string (this is what Perl does by default).

# In[16]:


print(re.match("c", "abcdef"))    # No match
print(re.search("^c", "abcdef"))  # No match, same as above


# In[15]:


print(re.search("c", "abcdef"))   # Match


# :::{note}
# 
# - `re.match` always matches at the beginning of the input string even if it is in the MULTILINE mode.
# 
# - `re.search` however, when in MULTILINE mode, is able to search at the beginning of every line if used in combination with `^`.
# 
# :::
# 
# 

# In[17]:


print(re.match('X', 'A\nB\nX', re.MULTILINE))  # No match
print(re.search('^X', 'A\nB\nX', re.MULTILINE))  # Match
print(re.search('^X', 'A\nB\nX')) # No match


# ### re.split()

# In[ ]:


text = """Ross McFluff: 834.345.1254 155 Elm Street

Ronald Heathmore: 892.345.3428 436 Finley Avenue
Frank Burger: 925.541.7625 662 South Dogwood Way


Heather Albrecht: 548.326.4584 919 Park Place"""


# In[64]:


# split text into lines
re.split(r'\n',text)


# In[63]:


re.split(r'\n+', text)


# In[ ]:


entries = re.split(r'\n+', text)


# In[66]:


[re.split(r'\s', entry) for entry in entries]


# In[69]:


[re.split(r':?\s', entry, maxsplit=3) for entry in entries]


# ### Text Munging

# - re.sub()

# In[70]:


text = '''Peter Piper picked a peck of pickled peppers
A peck of pickled peppers Peter Piper picked
If Peter Piper picked a peck of pickled peppers
Where’s the peck of pickled peppers Peter Piper picked?'''


# In[73]:


print(re.sub(r'[aeiou]','_', text))


# In[74]:


print(re.sub(r'([aeiou])',r'[\1]', text))


# In[82]:


American_dates = ["7/31/1976", "02.15.1970", "11-31-1986", "04/01.2020"]


# In[85]:


print(American_dates)
print([re.sub(r'(\d+)(\D)(\d+)(\D)(\d+)', r'\3\2\1\4\5', date) for date in American_dates])


# - In `re.sub(repl, string)`, the `repl` argument can be a function. If `repl` is a function, it is called for every non-overlapping occurrence of pattern. The function takes a single match object argument, and returns the replacement string.

# In[5]:


s = "This is a simple sentence."

pat_vowels = re.compile(r'[aeiou]')

def replaceVowels(m):
    c = m.group(0)
    c2 = ""
    if c in "ie":
        c2 = "F"
    else:
        c2 = "B"
    return c2
pat_vowels.sub(replaceVowels, s)


# ## References

# - [Python regular expression cheatsheet](https://learnbyexample.github.io/python-regex-cheatsheet/)
# - [Python official regular expression documentation](https://docs.python.org/3/library/re.html)
# - [Friedl, Jeffrey. Mastering Regular Expressions. 3rd ed., O’Reilly Media, 2009.](https://doc.lagout.org/programmation/Regular%20Expressions/Mastering%20Regular%20Expressions_%20Understand%20Your%20Data%20and%20Be%20More%20Productive%20%283rd%20ed.%29%20%5BFriedl%202006-08-18%5D.pdf)
