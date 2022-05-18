#!/usr/bin/env python
# coding: utf-8

# # Patterns and Constructions

# In[1]:


from nltk.chunk.regexp import tag_pattern2re_pattern
from nltk.chunk import RegexpParser
from nltk.corpus import brown
import nltk
tag_pattern2re_pattern('<DT>?<NN.*>+')


# In[2]:


chunker = RegexpParser('''
NP:
    {<DT><NN.*><.*>*<NN.*>}
    }<VB.*>{
''')

#sent = brown.tagged_sents()[10]
#chunker.parse(sent)

sent = 'The article looks like a work written by a foreigner.'
sent = nltk.pos_tag(nltk.word_tokenize(sent))
sent_ct = chunker.parse(sent)
print(sent_ct)


# In[3]:


sent_ct.productions()
#sent_ct.chomsky_normal_form()
sent_ct
sent_ct[0]


# In[4]:


print(type(sent_ct[0].label))
print(sent_ct[0])
print(type(sent_ct[0].leaves))
print(type(sent_ct[1]))
type(sent_ct[2])


# In[5]:


i=0
for subtree in sent_ct.subtrees():
    i=i+1
    print(str(i))
    print('label: {}'.format(subtree.label()))
    print(subtree)


# In[6]:


str(sent_ct)


# In[7]:


for subtree in sent_ct.subtrees(filter=lambda t: t.label().endswith("NP")):
    print(subtree)


# In[8]:


# write chunk rules
pat_chunker = RegexpParser('''
ADJ_AND_ADJ:
    {<JJ.*><CC><JJ.*>}
''')

for sent in brown.tagged_sents()[:500]:
    cur_t = pat_chunker.parse(sent)
    cur_pat = [pat for pat in cur_t.subtrees(filter=lambda t: t.label().startswith("ADJ_AND"))]
    if len(cur_pat)>0:
        print(cur_pat)
    


# In[9]:


nltk.help.upenn_tagset()


# ## Patterns from Raw-Text Corpus

# In[10]:


import nltk
from nltk.corpus import gutenberg



# In[11]:


gutenberg.fileids()


# In[12]:


alice_sents = [ ' '.join(sent) 
               for sent in gutenberg.sents(fileids='carroll-alice.txt')
              if len(sent)>=5]


# In[13]:


alice_sents[:5]


# In[14]:


import re

all_matches= [re.findall(r'(?:have|has)(?: [^s]+){0,2}[^\s]+(?:en|ed)', sent) for sent in alice_sents]


# In[15]:


print([m for m in all_matches if len(m)!=0])


# - The grouping parenthsis changes the behavior of `re.findall()`
# - With parenthesis, the regex engine automatically captures the matches in all the groups and return the results as a tuple.
# - Use `(?:...)` to create non-capturing gorups

# In[16]:


match = re.findall(r'(?:is|was) (?:\w+ing)', '''
Alice was beginning to get very tired of sitting by her sister on the bank , 
and of having nothing to do : once or twice she had peeped into the book 
her sister was reading , but it had no pictures or conversations in it , 
and what is the use of a book , thought Alice  without pictures or conversation ?
''')


if match:
    for m in match:
        print(m.strip())


# :::{important}
# It seems that when we use `re.findall()`, the matches returned would only be the capturing groups; but when we use `re.finditer()`, it would return the whole match strings as well as every section of the capturing groups.
# 
# I would prefer `re.finditer()`. Otherwise, `re.findall()` may need to specify the non-capturing group `(?:...)`.
# :::

# In[17]:


pat_perfect = re.compile(r'(is|was) (\w+ing)')
text = '''
Alice was beginning to get very tired of sitting by her sister on the bank , 
and of having nothing to do : once or twice she had peeped into the book 
her sister was reading , but it had no pictures or conversations in it , 
and what is the use of a book , thought Alice  without pictures or conversation ?
'''

pat_perfect_matches = pat_perfect.finditer(text)

if pat_perfect_matches:
    for m in pat_perfect_matches:
        print(m.group())

