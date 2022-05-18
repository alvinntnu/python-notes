# Praat TextGrid Data

- Module: [praatio](https://pypi.org/project/praatio/)
- Tutorial: [PraatIO-Doing Speech Analysis with Python](https://nbviewer.jupyter.org/github/timmahrt/praatIO/blob/master/tutorials/tutorial1_intro_to_praatio.ipynb)

from praatio import tgio
tg = tgio.openTextgrid('../../../Projects/MOST-Prosody/data/2014_di702_TextGrid_Alvin/di_001.TextGrid')

tg.tierDict

# get all intervals
tg.tierDict['PU'].entryList

tg.tierDict['Word'].entryList

tg.tierDict['Word'].find('我')



import pandas as pd
word_tier = tg.tierDict['Word']
pd.DataFrame([(start, end, label) for (start, end, label) in word_tier.entryList],
            columns = ['start','end','label'])


pu_tier = tg.tierDict['PU']
pd.DataFrame([(start, end, label) for (start, end, label) in pu_tier.entryList],
            columns = ['start','end','label'])
