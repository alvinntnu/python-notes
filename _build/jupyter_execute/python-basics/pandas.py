# Pandas

- Methods to deal with tabular data
- These methods are to replicate what `dplyr` in R is capable of
- The `statsmodels` can download R datasets from https://vincentarelbundock.github.io/Rdatasets/datasets.html

## Libraries

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
%matplotlib inline

## Importing/Exporting Data

Importing:

- `pd.read_csv(filename)`: From a CSV file
- `pd.read_table(filename)`: From a delimited text file (like TSV)
- `pd.read_excel(filename)`: From an Excel file
- `pd.read_sql(query, connection_object)`: Read from a SQL table/database
- `pd.read_json(json_string)`: Read from a JSON formatted string, URL or file.
- `pd.read_html(url)`: Parses an html URL, string or file and extracts tables to a list of dataframes
- `pd.read_clipboard()`: Takes the contents of your clipboard and passes it to read_table()
- `pd.DataFrame(dict)`: From a dict, keys for columns names, values for data as lists
- `pd.DataFrame(list of tuples)`: From a list, which includes the records of each row

Exporting:

- `df.to_csv(filename)` 
- `df.to_excel(filename)` 
- `df.to_sql(table_name, connection_object)` 
- `df.to_json(filename)`

DEMO_DATA_DIR = '../../../RepositoryData/data/titanic/'
iris = sm.datasets.get_rdataset('iris').data
titanic = pd.read_csv(DEMO_DATA_DIR+'train.csv')

x= [(1,2,3,4),
   (5,6,7,8),
   (9,10,11,12)]
pd.DataFrame(x,columns=["A","B","C","D"])

x = {"A":[1,2,3,4],
    "B":[5,6,7,8],
    "C":[9,10,11,12]}
pd.DataFrame(x)

```{note}
When you have data of the **columns**, use **dict**; when you have the data of the **rows**, use **list** as the source data structures of a data frame.
```

## Inspecting Data Frame

- `df.head(n)`: First n rows of the DataFrame
- `df.tail(n)`: Last n rows of the DataFrame
- `df.shape`: Number of rows and columns
- `df.info()`: Index, Datatype and Memory information
- `df.describe()`: Summary statistics for numerical columns
- `s.value_counts(dropna=False)`: View unique values and counts
- `df.apply(pd.Series.value_counts)`: Unique values and counts for all columns
- `df.columns`
- `df.index`
- `df.dtypes`
- `df.set_index('column_name')`: Set a column as the index 

iris.info()

iris.describe()

print(iris.shape)
iris.head(3)

titanic.tail(3)

iris['Species'].value_counts()

titanic.apply(pd.Series.value_counts)

print(iris.columns)
print(titanic.columns)
print(iris.index)

print(iris.dtypes)
print(titanic.dtypes)

## Basic Functions

## DataFrame attributes
iris.shape
iris.columns
iris.index
iris.info()
iris.describe()
iris.dtypes # check column data types


## Subsetting Data Frame

- `df[col]`: Returns column with label col as Series
- `df[[col1, col2]]`: Returns columns as a new DataFrame
- `s.iloc[0]`: Selection by position
- `s.loc['index_one']`: Selection by index
- `df.iloc[0,:]`: First row
- `df.iloc[0,0]`: First element of first column

iris.loc[:5, 'Species'] # first six rows of 'Species' column

iris.iloc[:5, 4] # same as above

## Exploration


How to perform the key functions provided in R `dplyr`?

- `dplyr` Key Verbs
    - `filter()`
    - `select()`
    - `mutate()`
    - `arrange()`
    - `summarize()`
    - `group_by()`

### NA Values


Functions to take care of `NA` values:
    
- `df.isnull()`
- `df.notnull()`
- `df.dropna()`: Drop rows with null values
- `df.dropna(axis=1)`: Drop columns with null values
- `df.dropna(axis=1, thresh=n)`: Drop all columns have less than n non-values
- `df.fillna(x)`: Replaces all null values with `x`
- `s.fillna(s.mean())`: Replace the null values of a Series with its mean score

- Quick check of the null values in each column

titanic.isnull().sum()

titanic.dropna(axis=1, thresh=600)

titanic.notnull().sum()

### Converting Data Types

- `s.astype(float)`: Convert a Series into a `float` type


iris.dtypes

iris['Species']=iris['Species'].astype('category')
iris.dtypes
#iris.value_counts(iris['Species']).plot.bar()

### Pandas-supported Data Types

![pandas-dtypes](../images/pandas-dtypes.png)

([source](https://pbpython.com/pandas_dtypes.html))


### Transformation


- `s.replace(X, Y)`

titanic.head()
titanic.value_counts(titanic['Survived']).plot.bar()
titanic.columns
titanic.groupby(['Sex','Pclass']).mean()
titanic[titanic['Age']<18].groupby(['Sex','Pclass']).mean()

###  `filter()`

## filter
iris[iris['Sepal.Length']>5]

```{note}
When there are more than one filtering condition, put the conditions in parentheses.
```

iris[(iris['Sepal.Length']>4) & (iris['Sepal.Width']>5)]

iris.query('`Sepal.Length`>5')

iris[(iris['Sepal.Length']>5) & (iris['Sepal.Width']>4)]

### `arrange()`

iris.sort_values(['Species','Sepal.Length'], ascending=[False,True])

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

### `apply()`, `mutate_if()`

- `df.apply(np.mean)`: Apply a function to all columns
- `df.apply(np.max,axis=1)`: Apply a function to each row

```{note}
When `apply()` functions to the data frame, the `axis=1` refers to row mutation and `axis=0` refers to column mutation. This is very counter-intuitive for R users.
```

iris.head(10)

iris[['Sepal.Width','Petal.Width']].apply(np.sum, axis=1).head(10)

### `group_by()` and `summarize()`

iris.groupby(by='Species').mean()

iris.filter(['Species','Sepal.Length']).groupby('Species').agg({'Sepal.Length':['mean','count','std']})


titanic.head()

titanic.groupby(['Pclass','Sex']).agg(np.sum)

titanic.pivot_table(index=['Pclass','Sex'], values=['Survived'], aggfunc=np.sum)

### `rename()`


iris
iris.columns

- Selective renaming column names

iris = iris.rename(columns={'Sepal.Length':'SLen'})
iris

- Massive renaming column names

iris.rename(columns=lambda x: 'XX'+x)

titanic.head(10)

titanic.set_index('Name').rename(index=lambda x:x.replace(' ',"_").upper())

## Join/Combine Data Frames

- `df1.append(df2)`: Add the rows in df1 to the end of df2 (columns should be identical) (`rbind()` in R)
- `pd.concat([df1, df2],axis=1)`: Add the columns in df1 to the end of df2 (rows should be identical) (`cbind()` in R)
- `df1.join(df2,on=col1,how='inner')`: SQL-style join the columns in df1 with the columns on df2 where the rows for col have identical values. 'how' can be one of 'left', 'right', 'outer', 'inner'



## Statistics

- `df.describe()`: Summary statistics for numerical columns
- `df.mean()`: Returns the mean of all columns
- `df.corr()`: Returns the correlation between columns in a DataFrame
- `df.count()`: Returns the number of non-null values in each DataFrame column
- `df.max()`: Returns the highest value in each column
- `df.min()`: Returns the lowest value in each column
- `df.median()`: Returns the median of each column
- `df.std()`: Returns the standard deviation of each column

titanic.count()

titanic.median()

## References

 - [Python for Data Analysis](https://www.amazon.com/gp/product/1491957662/ref=as_li_tl_nodl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1491957662&linkCode=as2&tag=ledoux-20&linkId=eff92247940c967299befaed855c580a)
 - [Python for Data Analysis GitHub](https://github.com/wesm/pydata-book)
 - [How to get sample datasets in Python](https://stackoverflow.com/questions/28417293/sample-datasets-in-pandas)


## Requirements

# %load get_modules.py
import pkg_resources
import types
def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, 
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

        # Some packages are weird and have different
        # imported names vs. system/pip names. Unfortunately,
        # there is no systematic way to get pip names from
        # a package's imported name. You'll have to add
        # exceptions to this list manually!
        poorly_named_packages = {
            "PIL": "Pillow",
            "sklearn": "scikit-learn"
        }
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]

        yield name
        
        
imports = list(set(get_imports()))

# The only way I found to get the version of the root package
# from only the name of the package is to cross-check the names 
# of installed packages vs. imported packages
requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name!="pip":
        requirements.append((m.project_name, m.version))

for r in requirements:
    print("{}=={}".format(*r))