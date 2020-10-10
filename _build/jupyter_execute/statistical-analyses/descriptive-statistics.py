# Descriptive Statistics

DEMO_DATA_ROOT = "../../../RepositoryData/data"

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

## Univariate Statistics

UHM = pd.read_table(DEMO_DATA_ROOT+"/gries_sflwr/_inputfiles/03-1_uh(m).csv")
UHM

UHM.value_counts(UHM['FILLER'])

UHM.value_counts(UHM['FILLER'], normalize=True)

def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x)+1)/len(x) # percentiles
    return(x,y)

ecdf(UHM.value_counts(UHM['FILLER']))