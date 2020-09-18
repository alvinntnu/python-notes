# Patterns and Constructions

from nltk.chunk.regexp import tag_pattern2re_pattern
from nltk.chunk import RegexpParser
from nltk.corpus import brown
import nltk
tag_pattern2re_pattern('<DT>?<NN.*>+')

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


sent_ct.productions()
#sent_ct.chomsky_normal_form()
sent_ct
sent_ct[0]

print(type(sent_ct[0].label))
print(sent_ct[0])
print(type(sent_ct[0].leaves))
print(type(sent_ct[1]))
type(sent_ct[2])

i=0
for subtree in sent_ct.subtrees():
    i=i+1
    print(str(i))
    print('label: {}'.format(subtree.label()))
    print(subtree)

str(sent_ct)

for subtree in sent_ct.subtrees(filter=lambda t: t.label().endswith("NP")):
    print(subtree)

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
    