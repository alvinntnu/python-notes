# Machine Learning with Sklearn -- Regression

## Data

import pandas as pd
import numpy as np

house = pd.read_csv("../../../RepositoryData/data/hands-on-ml/housing.csv")

house.head()

house.info()

house['ocean_proximity'].value_counts()

house.describe()

## Visualization

%matplotlib inline
import matplotlib.pyplot as plt
house.hist(bins=50, figsize=(20,15))
plt.show()

## Train-Test Split

### Simple random sampling

from sklearn.model_selection import train_test_split
train, test = train_test_split(house, test_size = 0.2, random_state=42)

len(train)

len(test)

### Stratified Random Sampling

The sampling should consider the income distributions.

house['income_cat'] = pd.cut(house['median_income'],
                            bins=[0,1.5, 3.0, 4.5, 6.0,np.inf],
                            labels=[1,2,3,4,5])

house['income_cat'].hist()

from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in splitter.split(house, house['income_cat']):
    train_strat = house.loc[train_index]
    test_strat = house.loc[test_index]

train_strat['income_cat'].value_counts()/len(train_strat)

test_strat['income_cat'].value_counts()/len(test_strat)

for set_ in (train_strat, test_strat):
    set_.drop("income_cat", axis = 1, inplace=True)

## Exploring Training Data

train_strat.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                s=train_strat["population"]/100, label="population",figsize=(10,7),
                c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True,)
plt.legend()

## size: population
## color: median house value

## Preparing Data for ML

- Wrap things in functions
    - Allow you to reproduce the same transformations easily on any dataset
    - Build a self-defined library of transformation functions
    - Allow you to apply the same transformation to new data in live system
    - Allow you to experiment with different transformations easily

housing = train_strat.drop("median_house_value", axis=1)
housing_labels = train_strat["median_house_value"].copy()
housing

### NA values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

imputer.statistics_

housing_num.median()

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
housing_tr

### Data Type Conversion

- Categorical to Ordinal

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories='auto')

ocean_proximity_cat = housing[["ocean_proximity"]]
ocean_proximity_cat.head(10)

ocean_proximity_ordinal = ordinal_encoder.fit_transform(ocean_proximity_cat)
ocean_proximity_ordinal[:10]

ordinal_encoder.categories_

pd.DataFrame(ocean_proximity_oridinal).value_counts()

- One-hot encoding

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder()
ocean_proximity_onehot = onehot_encoder.fit_transform(ocean_proximity_cat)

ocean_proximity_onehot

ocean_proximity_onehot.toarray()

### Feature Scaling

from sklearn.preprocessing import StandardScaler, MinMaxScaler

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

housing_num_minmax = minmax_scaler.fit_transform(housing_num)
housing_num_standard = standard_scaler.fit_transform(housing_num)

housing_num_minmax

housing_num_standard

### Transformation Pipelines

from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('standardizer', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

housing_num_tr

- Handle all columns all at once

from sklearn.compose import ColumnTransformer

num_columns = list(housing_num)
cat_columns = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_columns),
    ('cat', OneHotEncoder(), cat_columns)
])

housing_prepared = full_pipeline.fit_transform(housing)

```{important}
The output of the `full_pipeline` includes more columns than the original data frame `housing`. This is due to the one-hot encoding  transformation.
```

## Select and Train a Model

- Linear Regresssion

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmes = np.sqrt(lin_mse)
lin_rmes

- Decision Tree

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

tree_mse = mean_squared_error(housing_labels, tree_reg.predict(housing_prepared))
tree_rmes = np.sqrt(tree_mse)
tree_rmes

- Cross Validation to check over/under-fitting

from sklearn.model_selection import cross_val_score

- CV for Decision Tree

tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
tree_rmse_cv = np.sqrt(-tree_scores)
tree_rmse_cv

def display_cv_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())

display_cv_scores(tree_rmse_cv)

- CV for Linear Regression

linreg_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
linreg_rmse_cv = np.sqrt(-linreg_scores)
display_cv_scores(linreg_rmse_cv)

- CV for Random Forest

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                               scoring="neg_mean_squared_error", cv=10)
forest_rmse_cv=np.sqrt(-forest_scores)
display_cv_scores(forest_rmse_cv)

```{important}
- Try different ML algorithms first, before spending too much time tweaking the hyperparameters of one single algorithm.

- The first goal is to shortlist a few (2 to 5) promising models.
```