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
from spacy import displacy
# load language model
nlp_en = spacy.load('en') ## disable=["parser"]
# parse text 
doc = nlp_en('This is a sentence')


## Linguistic Features

- After we parse and tag a given text, we can extract token-level information:
    - Text: the original word text
    - Lemma: the base form of the word
    - POS: the simple universal POS tag
    - Tag: the detailed POS tag
    - Dep: Syntactic dependency
    - Shape: Word shape (capitalization, punc, digits)
    - is alpha
    - is stop
    
:::{admonition, dropdow, note}
For more information on POS tags, see spaCy (POS tag scheme documentation)[https://spacy.io/api/annotation#pos-tagging].
:::

# parts of speech tagging
for token in doc:
    print(((token.text, 
            token.lemma_, 
            token.pos_, 
            token.tag_,
            token.dep_,
            token.shape_,
            token.is_alpha,
            token.is_stop,
            )))

## Output in different ways
for token in doc:
    print('%s_%s' % (token.text, token.tag_))
    
out = ''
for token in doc:
    out = out + ' '+ '/'.join((token.text, token.tag_))
print(out)


## Check meaning of a POS tag
spacy.explain('VBZ')


## Visualization Linguistic Features

# Visualize
displacy.render(doc, style="dep")

options = {"compact": True, "bg": "#09a3d5",
           "color": "white", "font": "Source Sans Pro"}
displacy.render(doc, style="dep", options=options)


## longer paragraphs
text_long = '''In a presidency of unprecedented disruption and turmoil, Donald Trump's support has remained remarkably stable. That stability, paradoxically, points toward years of rising turbulence in American politics and life.
Trump's approval ratings and support in the presidential race against Democratic nominee Joe Biden have oscillated in a strikingly narrow range of around 40%-45% that appears largely immune to both good news -- the long economic boom during his presidency's first years -- and bad -- impeachment, the worst pandemic in more than a century, revelations that he's disparaged military service and blunt warnings that he is unfit for the job from former senior officials in his own government. Perhaps the newest disclosures, in the upcoming book from Bob Woodward, that Trump knew the coronavirus was far more dangerous than the common flu even as he told Americans precisely the opposite, will break this pattern, but most political strategists in both parties are skeptical that it will.
'''
## parse the texts
doc2 = nlp_en(text_long)
## get the sentences of the doc2
sentence_spans = list(doc2.sents)
options = {"compact": True, "bg": "#09a3d5",
           "color": "white", "font": "Source Sans Pro",
          "distance": 120}
displacy.render(sentence_spans, style="dep", options=options)

colors = {"ORG": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
options = {"ents": ["ORG"], "colors": colors}


displacy.render(sentence_spans[0], style="ent")

## NP Chunking

for c in doc2.noun_chunks:
    print(c.text, c.root.text, c.root.dep_, c.root.head.text)

## Named Entity Recognition

- Text: original entity text
- Start: index of start of entity in the Doc
- End: index of end of entity in the Doc
- Label: Entity label, type


for ent in doc2.ents:
    print(ent.text,
         ent.start_char,
         ent.end_char,
         ent.label_)