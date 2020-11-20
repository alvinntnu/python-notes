# WordNet

WordNet is a lexical database for the English language, where word senses are connected as a systematic lexical network.

## Import

from nltk.corpus import wordnet

## Synsets

A `synset` has several attributes, which can be extracted via its defined methods:

- `synset.name()`
- `synset.definition()`
- `synset.hypernyms()`
- `synset.hyponyms()`
- `synset.hypernym_path()`
- `synset.pos()`

syn = wordnet.synsets('walk', pos='v')[0]
print(syn.name())
print(syn.definition())


syn.examples()

syn.hypernyms()

syn.hypernyms()[0].hyponyms()

syn.hypernym_paths()

syn.pos()

## Lemmas 

A `synset` may coreespond to more than one lemma.

syn = wordnet.synsets('walk', pos='n')[0]
print(syn.lemmas())

Check the lemma names.

for l in syn.lemmas():
    print(l.name())

## Synonyms

synonyms = []
for s in wordnet.synsets('run', pos='v'):
    for l in s.lemmas():
        synonyms.append(l.name())
print(len(synonyms))
print(len(set(synonyms)))

print(set(synonyms))

## Antonyms

Some lemmas have antonyms.

The following examples show how to find the antonyms of `good` for its two different senses, `good.n.02` and `good.a.01`.

syn1 = wordnet.synset('good.n.02')
syn1.definition()

ant1 = syn1.lemmas()[0].antonyms()[0]

ant1.synset().definition()

ant1.synset().examples()

syn2 = wordnet.synset('good.a.01')
syn2.definition()

ant2 = syn2.lemmas()[0].antonyms()[0]

ant2.synset().definition()

ant2.synset().examples()

## Wordnet Synset Similarity

With a semantic network, we can also compute the semantic similarty between two synsets based on their distance on the tree. 

In particular, this is possible cause all synsets are organized in a hypernym tree.

The recommended distance metric is Wu-Palmer Similarity (i.e., `synset.wup_similarity()`)

s1 = wordnet.synset('walk.v.01')
s2 = wordnet.synset('run.v.01')
s3 = wordnet.synset('toddle.v.01')

s1.wup_similarity(s2)

s1.wup_similarity(s3)

s1.common_hypernyms(s3)

s1.common_hypernyms(s2)

Two more metrics for lexical semilarity:

- `synset.path_similarity()`: Path Similarity
- `synset.lch_similarity()`: Leacock Chordorow Similarity