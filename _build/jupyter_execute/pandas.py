# Pandas

- Recommended Readings
    - [Python for Data Analysis](https://www.amazon.com/gp/product/1491957662/ref=as_li_tl_nodl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1491957662&linkCode=as2&tag=ledoux-20&linkId=eff92247940c967299befaed855c580a)
    - [Python for Data Analysis GitHub](https://github.com/wesm/pydata-book)
    - [How to get sample datasets in Python](https://stackoverflow.com/questions/28417293/sample-datasets-in-pandas)
- Alternative methods in Python to deal with data exploration and manipulation
- These methods are to replicate what `dplyr` in R is capable of
- The `statsmodels` can download R datasets from https://vincentarelbundock.github.io/Rdatasets/datasets.html

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
%matplotlib inline

iris = sm.datasets.get_rdataset('iris').data
titanic = pd.read_csv('../../../Corpus/StatsDatasets/titanic-train.csv')

## Quick way to access R datasets
pd.DataFrame.head(iris)

## DataFrame attributes
iris.shape
iris.columns
iris.index
iris.info()
iris.describe()
iris.dtypes # check column data types


## `dplyr` Key Verbs

- `filter()`
- `select()`
- `mutate()`
- `arrange()`
- `summarize()`
- `group_by()`

iris.isnull().sum()
iris['Species']=iris['Species'].astype('category')
iris.dtypes
#iris.value_counts(iris['Species']).plot.bar()

titanic.head()
titanic.value_counts(titanic['Survived']).plot.bar()
titanic.columns
titanic.groupby(['Sex','Pclass']).mean()
titanic[titanic['Age']<18].groupby(['Sex','Pclass']).mean()

###  `filter()`

## filter
iris[iris['Sepal.Length']>5]

iris.query('`Sepal.Length`>5')

iris[(iris['Sepal.Length']>5) & (iris['Sepal.Width']>4)]

### `select()`

## select
iris[['Sepal.Length', 'Species']]

## deselect columns
iris.drop(['Sepal.Length'], axis=1).head()

iris.filter(['Species','Sepal.Length'])

iris[['Species','Sepal.Length']]

## extract one particular column
sepal_length = iris['Sepal.Length']
type(sepal_length)

### `mutate()`

## mutate
iris['Species_new'] = iris['Species'].apply(lambda x: len(x))
iris['Species_initial'] = iris['Species'].apply(lambda x: x[:2].upper())
iris

## mutate alternative 2
iris.assign(Specias_initial2 = iris['Species'].apply(lambda x: x.upper()))

### `group_by()` and `summarize()`

iris.groupby(by='Species').mean()

iris.filter(['Species','Sepal.Length']).groupby('Species').agg({'Sepal.Length':['mean','count','std']})


### `rename()`


iris
iris.columns

iris = iris.rename(columns={'Sepal.Length':'SLen'})
iris