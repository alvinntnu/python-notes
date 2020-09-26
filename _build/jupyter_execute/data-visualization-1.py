# Data Visualization I

![](../image/seaborn-func.png)
(source: [documentation](https://seaborn.pydata.org/tutorial/function_overview.html)

```{note}
Still like the vidualization tools in R better.
Seaborn is not intuitive, esp. for long-wide conversion of data frames.
```

## Matlibplot

- [Python Graph Gallery](https://python-graph-gallery.com/)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from plotnine.data import mpg, mtcars

# mtcars.head()

# Make a data frame
df=pd.DataFrame({'x': range(1,11), 'y1': np.random.randn(10), 'y2': np.random.randn(10)+range(1,11), 'y3': np.random.randn(10)+range(11,21), 'y4': np.random.randn(10)+range(6,16), 'y5': np.random.randn(10)+range(4,14)+(0,0,0,0,0,0,0,-3,-8,-6), 'y6': np.random.randn(10)+range(2,12), 'y7': np.random.randn(10)+range(5,15), 'y8': np.random.randn(10)+range(4,14), 'y9': np.random.randn(10)+range(4,14), 'y10': np.random.randn(10)+range(2,12) })

df.head()

# style
plt.style.use('seaborn-darkgrid')
 
# create a color palette
palette = plt.get_cmap('Set1')
 
# multiple line plot
num=0
for column in df.drop('x', axis=1):
    num+=1
    plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
 
# Add legend
plt.legend(loc=2, ncol=2)
 
# Add titles
plt.title("A (bad) Spaghetti plot", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Time")
plt.ylabel("Score")

# Make a data frame
df=pd.DataFrame({'x': range(1,11), 'y1': np.random.randn(10), 'y2': np.random.randn(10)+range(1,11), 'y3': np.random.randn(10)+range(11,21), 'y4': np.random.randn(10)+range(6,16), 'y5': np.random.randn(10)+range(4,14)+(0,0,0,0,0,0,0,-3,-8,-6), 'y6': np.random.randn(10)+range(2,12), 'y7': np.random.randn(10)+range(5,15), 'y8': np.random.randn(10)+range(4,14), 'y9': np.random.randn(10)+range(4,14) })
 
# Initialize the figure
plt.style.use('seaborn-darkgrid')
 
# create a color palette
palette = plt.get_cmap('Set1')

df.head()

# multiple line plot
num=0
for column in df.drop('x', axis=1):
    num+=1
    # Find the right spot on the plot
    plt.subplot(3,3, num)
    # Plot the lineplot
    plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1.9, alpha=0.9, label=column)
    # Same limits for everybody!
    plt.xlim(0,10)
    plt.ylim(-2,22)
    # Not ticks everywhere
    if num in range(7) :
        plt.tick_params(labelbottom='off')
    if num not in [1,4,7] :
        plt.tick_params(labelleft='off')
    # Add title
    plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )
# general title
plt.suptitle("How the 9 students improved\nthese past few days?", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)

# Axis title
plt.text(0.5, 0.02, 'Time', ha='center', va='center')
plt.text(0.06, 0.5, 'Note', ha='center', va='center', rotation='vertical')

## Seaborn Module

## Two Types of Functions

- Figure-level functions (Generic)
- Axex-level functions (Specific)

import seaborn as sns
%matplotlib inline
sns.set(style='darkgrid')

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