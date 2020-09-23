## Data Visualization Using Seaborn

![](../image/seaborn-func.png)
(source: [documentation](https://seaborn.pydata.org/tutorial/function_overview.html)

```{note}
Still like the vidualization tools in R better.
Seaborn is not intuitive, esp. for long-wide conversion of data frames.
```

## Two Types of Functions

- Figure-level functions (Generic)
- Axex-level functions (Specific)

import seaborn as sns
%matplotlib inline

penguins = sns.load_dataset('penguins')
penguins.head()

# histogram
sns.histplot(data=penguins, x='flipper_length_mm', hue='species', multiple="stack")

sns.displot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")

sns.displot(data=penguins, x="flipper_length_mm", hue="species", col="species")

## kernel density plot
sns.kdeplot(data=penguins, x='flipper_length_mm', hue='species', multiple="stack")

sns.displot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack", kind="kde")


tips = sns.load_dataset("tips")

g = sns.relplot(data=tips, x="total_bill", y="tip")
g.ax.axline(xy1=(10,2), slope=.2, color="b", dashes=(5,2))

g = sns.relplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", col="sex")
g.set_axis_labels("Flipper length (mm)", "Bill length (mm)")

sns.catplot(data=penguins, x='species', y='flipper_length_mm', kind="box")

- `jointplot()`: plots the relationship or joint distribution of two variables while adding marginal axes that show the univariate distribution of each one separately

sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species")

- `pairplot()`: visualizes every pairwise combination of variables simultaneously in a data frame


sns.pairplot(data=penguins, hue="species")

## Long-format vs. Wide-format Data

flights = sns.load_dataset("flights")
flights.head()

sns.relplot(data=flights, x="year", y="passengers", hue="month", kind="line")


sns.relplot(data=flights, x="month", y="passengers", hue="year", kind="line")

flights_wide = flights.pivot(index="year", columns="month", values="passengers")
flights_wide.head()


print(type(flights_wide))

sns.catplot(data=flights_wide, kind="box")