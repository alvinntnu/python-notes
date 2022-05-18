#!/usr/bin/env python
# coding: utf-8

# # CHILDES Corpus
# 
# 

# - This section includes two methods to process CHILDES data: `nltk` and `pylangacq`.
# - Good for language acquisition analysis
# - NLTK can deal with the xml format of the CHILDES corpus
# - CHILDES xml is available at [https://childes.talkbank.org/data-xml/](https://childes.talkbank.org/data-xml/)
# 

# In[1]:


DEMO_DATA_ROOT = "../../../RepositoryData/data"


# In[2]:


import nltk
from nltk.corpus.reader import CHILDESCorpusReader
r = CHILDESCorpusReader(DEMO_DATA_ROOT+'/CHILDES_Chang1_xml', '.*.xml')


# In[3]:


r.fileids()


# In[4]:


# print basic profile for each xml
for f in r.fileids()[:5]:
    cur_corpus = r.corpus(f)[0]
    print(cur_corpus['Corpus'],
          cur_corpus['PID'],
         cur_corpus['ActivityType'],
         cur_corpus['Date'])
    print("Num of Words: {}".format(len(r.words(f))))
    print("Num of Sents: {}".format(len(r.sents(f))))


# In[5]:


# participants
r.participants(fileids=r.fileids()[10])[0]# first file participants


# In[6]:


all_speakers = r.participants()

for speakers_cur_file in all_speakers[:5]:
    print("====")
    for spid in speakers_cur_file.keys():
        cur_spid_data = speakers_cur_file[spid]
        print(spid, ": ", [(param, cur_spid_data[param]) for param in cur_spid_data.keys()] )


# In[7]:


r.words('01.xml')
print(r.sents('01.xml', speaker='EXP'))
print(r.sents('01.xml', speaker='CHI')) # replace=T, stem=True


# In[8]:


# age
r.age()
r.age(month=True)
r.age(fileids='01.xml', month=True)


# In[9]:


# MLU
r.MLU(fileids='01.xml')


# In[10]:


[(age, mlu)  
for f in r.fileids()
for age in r.age(fileids = f, month=True)
for mlu in r.MLU(fileids = f)
]


# In[11]:


import pandas

age_mlu_data = pandas.DataFrame([(age, mlu)  
for f in r.fileids()
for age in r.age(fileids = f, month=True)
for mlu in r.MLU(fileids = f)
], columns=['Age','MLU'])


# In[12]:


get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[13]:


get_ipython().run_cell_magic('R', '', 'library(ggplot2)\nlibrary(dplyr)\n')


# In[14]:


get_ipython().run_cell_magic('R', '-i age_mlu_data', 'age_mlu_data %>%\nggplot(aes(Age, MLU)) +\ngeom_point(size=2) +\ngeom_smooth(method="lm") +\nlabs(x="Child Age(Months)",y="Mean Length of Utterances (MLU)")\n')


# ## CHA file
# 
# - Fantastic package for CHA files: [PyLangAcq](http://pylangacq.org/)

# In[15]:


import pylangacq as pla
pla.__version__  # show version number


# In[16]:


nccu = pla.read_chat(DEMO_DATA_ROOT+'/CHILDES_NCCU/transcript/*.cha')


# In[17]:


nccu.number_of_files()


# In[18]:


print('Corpus Size:', len(nccu.words()))


# In[19]:


all_headers= nccu.headers()
#all_headers[list(all_headers.keys())[0]]
list(all_headers.items())[0]


# In[20]:


nccu.word_frequency().most_common(5)
nccu.word_ngrams(n=3).most_common(10)


# In[21]:


for line in [' '.join(sent) for sent in nccu.sents()[:10]]:
    print(line)

