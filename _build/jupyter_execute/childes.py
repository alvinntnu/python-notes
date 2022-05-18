# CHILDES Corpus

- NLTK can deal with the xml format of the CHILDES corpus
- CHILDES xml is available at [https://childes.talkbank.org/data-xml/](https://childes.talkbank.org/data-xml/)


import nltk
from nltk.corpus.reader import CHILDESCorpusReader
r = CHILDESCorpusReader('../../../Corpus/CHILDES/Chinese/Chang1_xml/', '.*.xml')

r.fileids()

# print basic profile for each xml
for f in r.fileids()[:5]:
    cur_corpus = r.corpus(f)[0]
    print(cur_corpus['Corpus'],
          cur_corpus['PID'],
         cur_corpus['ActivityType'],
         cur_corpus['Date'])
    print("Num of Words: {}".format(len(r.words(f))))
    print("Num of Sents: {}".format(len(r.sents(f))))

# participants
r.participants(fileids=r.fileids()[10])[0]# first file participants

all_speakers = r.participants()

for speakers_cur_file in all_speakers[:5]:
    print("====")
    for spid in speakers_cur_file.keys():
        cur_spid_data = speakers_cur_file[spid]
        print(spid, ": ", [(param, cur_spid_data[param]) for param in cur_spid_data.keys()] )

r.words('01.xml')
print(r.sents('01.xml', speaker='EXP'))
print(r.sents('01.xml', speaker='CHI')) # replace=T, stem=True

# age
r.age()
r.age(month=True)
r.age(fileids='01.xml', month=True)

# MLU
r.MLU(fileids='01.xml')

[(age, mlu)  
for f in r.fileids()
for age in r.age(fileids = f, month=True)
for mlu in r.MLU(fileids = f)
]

import pandas

age_mlu_data = pandas.DataFrame([(age, mlu)  
for f in r.fileids()
for age in r.age(fileids = f, month=True)
for mlu in r.MLU(fileids = f)
], columns=['Age','MLU'])

%load_ext rpy2.ipython

%%R
library(ggplot2)
library(dplyr)

%%R -i age_mlu_data
age_mlu_data %>%
ggplot(aes(Age, MLU)) +
geom_point(size=2) +
geom_smooth(method="lm") +
labs(x="Child Age(Months)",y="Mean Length of Utterances (MLU)")

## CHA file

- Fantastic package for CHA files: [PyLangAcq](http://pylangacq.org/)

import pylangacq as pla
pylangacq.__version__  # show version number

nccu = pla.read_chat('../../../Corpus/NCCUTaiwanMandarin/transcript/*.cha')

nccu.number_of_files()

print('Corpus Size:', len(nccu.words()))

all_headers= nccu.headers()
#all_headers[list(all_headers.keys())[0]]
list(all_headers.items())[0]

nccu.word_frequency().most_common(5)
nccu.word_ngrams(n=3).most_common(10)

for line in [' '.join(sent) for sent in nccu.sents()[:10]]:
    print(line)

