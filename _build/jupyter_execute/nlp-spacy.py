# Natural Language Processing (spaCy)

## Installation

```
# Install package
## In terminal:
!pip install spacy

## Download language model for Chinese and English
!spacy download en
!spacy download zh
```

import spacy
nlp_en = spacy.load('en')

doc = nlp_en('This is a sentence')


## POS Tagging

# parts of speech tagging
for token in doc:
    print(((token.text, token.pos_, token.tag_)))

for token in doc:
    print('%s_%s' % (token.text, token.tag_))


out = ''
for token in doc:
    out = out + ' '+ '/'.join((token.text, token.tag_))
print(out)

## Chunking

for c in doc.noun_chunks:
    print(c.text, c.root.text, c.root.dep_, c.root.head.text)