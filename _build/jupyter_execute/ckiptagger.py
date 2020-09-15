# Chinese Word Segmentation (ckiptagger)

The current state-of-art Chinese segmenter for Taiwan Mandarin available is probably the [CKIP tagger](https://github.com/ckiplab/ckiptagger), created by the [Chinese Knowledge and Information Processing (CKIP)](https://ckip.iis.sinica.edu.tw/) group at the Academia Sinica.

The `ckiptagger` is released as a python module. In this chpater, I will demonstrate how to use the module for Chinese word segmentation but in an R environment, i.e., how to integrate Python modules in R coherently to perform complex tasks.

## Installation

Because `ckiptagger` is built in python, we need to have python installed in our working environment. Please install the following applications on your own before you start:

- [Anaconda + Python 3.6+](https://www.anaconda.com/distribution/)
- `ckiptagger` module in Python (Please install the module using the `Anaconda Navigator` or `pip install` in the terminal)

(**Please consult the github of the [`ckiptagger`](https://github.com/ckiplab/ckiptagger) for more details on installation.**)

```{note}
For some reasons, the module `ckiptagger` may not be found in the base channel. In `Anaconda Navigator`, if you cannot find this module, please add specifically the following channel to the environment so that your Anaconda can find `ckiptagger` module:

`https://conda.anaconda.org/roccqqck`
```

## Download the Model Files

All NLP applications have their models behind their fancy performances. To use the tagger provided in `ckiptagger`, we need to download their pre-trained model files. 

Please go to the [github of CKIP tagger](https://github.com/ckiplab/ckiptagger) to download the model files, which is provided as a zipped file. (The file is very big. It takes a while.)

After you download the zipped file, unzip it under your working directory to the `data/` directory.

## Word Segmentation

Before we proceed, please check if you have everything ready (The following includes the versions of the modules used for this session):

- Anaconda + Python 3.6+ (`Python 3.6.10`)
- Python modules: `ckiptagger` (`ckiptagger 0.1.1` + `tensorflow 1.13.1`)
- CKIP model files under your working directory `./data`

If yes, then we are ready to go.


## Creating Conda Environment for `ckiptagger`

I would suggest to install all necessary Python modules in a conda environment for easier use. 

In the following demonstration, I assume that you have created a conda environment `ckiptagger`, where all the necessary modules (i.e., `ckiptagger`, `tensorflow`) have been pip-installed.

```
# isntsall in terminal
## create new env
conda create --name ckiptagger python=3.6
conda activate ckiptagger
pip install -U ckiptagger
## AND INSTALL EVERYTHING NEEDED FOR YOUR PROJECR

## deactivate env when you are done
conda deactivate
```

## Segmenting Texts

The initialized word segmenter object, `ws()`, can tokenize any input **character vectors** into a list of **word vectors** of the same size.

from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

# Set Parameter Path
MODEL_PATH = '../../../NTNU/CorpusLinguistics/CorpusLinguistics_bookdown/data/'
#'/Users/Alvin/Dropbox/NTNU/CorpusLinguistics/CorpusLinguistics_bookdown/data/'
## Loading model
#ws = WS('/Users/Alvin/Dropbox/NTNU/CorpusLinguistics/CorpusLinguistics_bookdown/data/')
ws = WS(MODEL_PATH)
#ws = WS('../../../NTNU/CorpusLinguistics/CorpusLinguistics_bookdown/data/')
pos = POS(MODEL_PATH)
ner = NER(MODEL_PATH)

## Raw text corpus 
sentence_list = ['傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。',
              '美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。',
              '土地公有政策?？還是土地婆有政策。',
              '… 你確定嗎… 不要再騙了……他來亂的啦',
              '最多容納59,000個人,或5.9萬人,再多就不行了.這是環評的結論.',
              '科長說:1,坪數對人數為1:3。2,可以再增加。']
    ## other parameters
    # sentence_segmentation = True, # To consider delimiters
    # segment_delimiter_set = {",", "。", ":", "?", "!", ";"}), # This is the defualt set of delimiters
    # recommend_dictionary = dictionary1, # words in this dictionary are encouraged
    # coerce_dictionary = dictionary2, # words in this dictionary are forced

word_list = ws(sentence_list)
pos_list = pos(word_list)
entity_list = ner(word_list, pos_list)
    

def print_word_pos_sentence(word_sentence, pos_sentence):
    assert len(word_sentence) == len(pos_sentence)
    for word, pos in zip(word_sentence, pos_sentence):
        print(f"{word}({pos})", end="\u3000")
    print()
    return
    
for i, sentence in enumerate(sentence_list):
    print()
    print(f"'{sentence}'")
    print_word_pos_sentence(word_list[i],  pos_list[i])
    for entity in sorted(entity_list[i]):
        print(entity)


## Define Own Dictionary

The performance of Chinese word segmenter depends highly on the dictionary. Texts in different disciplines may have very domain-specific vocabulary. To prioritize a set of words in a dictionary, we can further ensure the accuracy of the word segmentation.

To create a dictionary for `ckiptagger`:


word_to_weight = {
    "土地公": 1,
    "土地婆": 1,
    "公有": 2,
    "": 1,
    "來亂的": "啦",
    "緯來體育台": 1,
}
dictionary = construct_dictionary(word_to_weight)
print(dictionary)

word_list_2 = ws(sentence_list,
                recommend_dictionary=dictionary)
print(word_list)
print(word_list_2)

## Convert ckiptagger output into a Data Frame?