# Lexical Bundles

This section talks about how to identify recurring multiword sequences from texts, which has received a lot of attention in recent years in language studies.

## Loading libraries

from nltk.corpus import reuters
from nltk import ngrams
from collections import Counter, defaultdict
import re

## Corpus Data

In this demonstration, we use the `reuters` corpus as our data source, which has been made available in the `nltk`.

## A quick look at the first five sentences
print([' '.join(sent) for sent in reuters.sents()[:5]])

## Lexical Bundles

Lexical bundles refer to any contiguous multiword sequences from the texts. Normally, research on lexical bundles examine the multiword sequences of sizes from four- to seven-word sequences.

The idea of lexical bundles is essentially the ngrams in NLP, which `N` refers to the size of the multiword sequence.

To extract a meaningful set of lexical bundles, we need to consider at least two important distributional criteria:

- **Frequency** of the bundle: how often does the sequence occur in the entire corpus?
- **Range** of the bundle: in how many different texts/documents does the sequence occur in the entire corpus?

## Number of documents in `reuters`
len(reuters.fileids())

# Create a placeholder for 4gram bundles statistics
bundles_4 = defaultdict(lambda: defaultdict(lambda: 0))
bundles_range = defaultdict(lambda: defaultdict(lambda: 0))

[n for n in ngrams(reuters.sents()[1],n=4)]

%%time
# Count frequency of co-occurance  
for fid in reuters.fileids():
    temp = defaultdict(lambda: defaultdict(lambda: 0))
    for sentence in reuters.sents(fileids=fid):
        for w1, w2, w3, w4 in ngrams(sentence, n=4, pad_right=False, pad_left=False):
            ## filter
            if re.match(r'\w+',w1) and re.match(r'\w+',w2) and re.match(r'\w+',w3) and re.match(r'\w+', w4):
                bundles_4[(w1, w2, w3)][w4] += 1
                temp[(w1, w2, w3)][w4] += 1
    # range value
    for key, value in temp.items():
        for k in value.keys():
            bundles_range[key][k] +=1

list(bundles_4.items())[:5]

list(bundles_range.items())[:5]

## Convert to data frames

- For more intuitive reading of the bundles data, we can create a data frame with the distributional information of each bundle type.
- Most importantly, we can filter and sort our bundle data nicely and easily with the functionality provided with the data frame.

Create three lists:

- `w1_w2_w3`: the first three words in the bundle
- `w4`: the last word in the bundle
- `freq`: freq of the bundle
- `range`: range of the bundle

%%time
import pandas as pd

w1_w2_w3 = []
w4 = []
freq = []
rangev = []
for _w123 in bundles_4.keys():
    for _w4 in bundles_4[_w123].keys():
        w1_w2_w3.append('_'.join(_w123))
        w4.append(_w4)
        freq.append(bundles_4[_w123][_w4])
        rangev.append(bundles_range[_w123][_w4])
        

Check the lengths of the four lists before combining them into a data frame.

print(len(w1_w2_w3))
print(len(w4))
print(len(freq))

Create the bundle data frame.

bundles_df =pd.DataFrame(list(zip(w1_w2_w3, w4, freq, rangev)),
                        columns=['w123','w4','freq','range'])
bundles_df.head()


Filter bundles whose `range` >= 10 and arrange the data frame according to bundles' `range` values.

bundles_df[(bundles_df['range']>=10)].sort_values(['range'], ascending=[False]).head(20)

Identify bundles with w4 being either `in` or `to`.

bundles_df[(bundles_df['range']>=10) & (bundles_df['w4'].isin(['in','to']))].sort_values(['range'], ascending=[False]).head(20)

## Restructure dictionary

# ## filter and sort

# ## remove ngrams with non-word characters
# bundles_4_2 = {(w1,w2,w3):value for (w1,w2,w3), value in bundles_4.items() if 
#                re.match(r'\w+',w1) and re.match(r'\w+',w2) and re.match(r'\w+',w3)}

# print(len(bundles_4))
# print(len(bundle_4_2))


# ## remove ngrams whose freq < 5 and w4 with non-word characters
# bundles_4_3 = {}
# for w1_w2_w3 in bundles_4_2:
#     bundles_4_3[w1_w2_w3] = {w4:v for w4, v in bundles_4[w1_w2_w3].items() if v >= 5 and re.match(r'\w+',w4)}

# ## clean up dictionary
# bundles_4_3 = {key:value for key,value in bundles_4_3.items() if len(value)!=0}
    
# print(list(bundles_4_3.items())[:5])
# print(len(bundles_4_3))

#  # From raw frequencies to forward transitional probabilities
# for w1_w2_w3 in bundles_4:
#     total_count = float(sum(bundles_4[w1_w2_w3].values()))
#     for w4 in bundles_4[w1_w2_w3]:
#         bundles_4[w1_w2_w3][w4] /= total_count

# ## flatten the dictionary
# bundles_4_4 = {}
# for w1_w2_w3 in bundles_4_3:
#     for w4 in bundles_4_3[w1_w2_w3]:
#         ngram = list(w1_w2_w3)+[w4]
#         bundles_4_4[tuple(ngram)] = bundles_4_3[w1_w2_w3][w4]

# sorted(bundles_4_4.items(), key=lambda x:x[1],reverse=True)