# Analytic Statistics

- Include common analytic statistical analyses
- Still, R is better for these tasks.

import numpy as np
import pandas as pd
import scipy.stats as stats

DEMO_DATA_ROOT = "../../../RepositoryData/data"

## Kruskal Test

- Two independent sample means

hedges = pd.read_table(DEMO_DATA_ROOT + "/gries_sflwr/_inputfiles/04-1-2-1_hedges.csv")
hedges.head()

u_statistic, p = stats.ks_2samp(hedges[hedges['SEX']=="M"]['HEDGES'],hedges[hedges['SEX']=="F"]['HEDGES'] )
print(u_statistic, '\n', p)

## Chi-square

data = np.array([[85, 65],
                 [100,147]])
data

V, p, df, expected = stats.chi2_contingency(data, correction=False)
print("""
Chi-square value = {}
dF = {}
p = {}
""".format(V, df, p))

## McNear Test

- One dependent variable (categorical)
- dependent samples

data = pd.read_table(DEMO_DATA_ROOT + "/gries_sflwr/_inputfiles/04-1-2-3_accjudg.csv")
data.head()

from statsmodels.sandbox.stats.runs import mcnemar

crosstab = pd.crosstab(data['BEFORE'],data['AFTER'])
x2, p = mcnemar(crosstab, correction=False)
print('Chi-square={}, p = {}'.format(x2, p))

## Independent *t*-test

vowels = pd.read_table(DEMO_DATA_ROOT + "/gries_sflwr/_inputfiles/04-3-2-1_f1-freq.csv")
vowels.head()

t, p = stats.ttest_ind(vowels[vowels['SEX']=='M']['HZ_F1'], vowels[vowels['SEX']=='F']['HZ_F1'])
print("t-score={}, p={}".format(t,p))

