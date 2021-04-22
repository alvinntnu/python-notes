# Data Visualization I

## Preparing Datasets

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib


# Make a data frame
df=pd.DataFrame({'x': range(1,11), 'y1': np.random.randn(10), 'y2': np.random.randn(10)+range(1,11), 'y3': np.random.randn(10)+range(11,21), 'y4': np.random.randn(10)+range(6,16), 'y5': np.random.randn(10)+range(4,14)+(0,0,0,0,0,0,0,-3,-8,-6), 'y6': np.random.randn(10)+range(2,12), 'y7': np.random.randn(10)+range(5,15), 'y8': np.random.randn(10)+range(4,14), 'y9': np.random.randn(10)+range(4,14), 'y10': np.random.randn(10)+range(2,12) })

df['x']=pd.Categorical(df['x'])
print(df.dtypes)
df.head(10)


## Matlibplot


### Resolution

- We can increase the dpi of the matplotlib parameters to get image of higher resolution in notebook
- The dpi setting has to GO before the magic inline command because the magic inline commened resets the dpi to default



## Change DPI for higher resolution in notebook
%matplotlib inline

matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 150
# Change DPI when saving graphs in files
# matplotlib.rc("savefig", dpi=dpi)


### Matplotlib Style

# available style
print(plt.style.available)

# choose one style
plt.style.use('fivethirtyeight')

### Matplotlib Chinese Issues

## Setting Chinese Fonts
## Permanent Setting Version
plt.rcParams['font.sans-serif']=["PingFang HK"]
plt.rcParams['axes.unicode_minus']= False

### Plotting



## Simple X and Y
plt.plot(df['x'], df['y1'])
plt.show()

## Simple X and two Y's
plt.plot(df['x'], df['y1'])
plt.plot(df['x'], df['y2'])
plt.show()

## Adding legends
## Simple X and two Y's
plt.plot(df['x'], df['y1'], label="資料1")
plt.plot(df['x'], df['y2'], label="資料2")
plt.legend()
plt.tight_layout()
plt.show()

## Save graphs
## Simple X and two Y's
plt.plot(df['x'], df['y1'], label="資料1")
plt.plot(df['x'], df['y2'], label="資料2")
plt.legend()
plt.tight_layout()
plt.savefig('plot.png')
plt.show()

## Add x/y labels and title
## Simple X and two Y's
plt.plot(df['x'], df['y1'], label="資料1")
plt.plot(df['x'], df['y2'], label="資料2")
plt.legend()
plt.xlabel("X軸")
plt.ylabel("Y軸")
plt.title("漂亮的圖")
plt.tight_layout()
plt.show()

### Bar Plots

## Normal bar plot
plt.bar(df['x'], df['y3'])
plt.show()

## Sort bars according to values

df_sorted = df.sort_values(['y3','y2'], ascending=True)
print(df_sorted.dtypes)
df_sorted

plt.bar('x', 'y3', data=df_sorted)
plt.show()

## Horizontal Bars
plt.bar('x', 'y4', data=df.sort_values('y4'))
plt.tight_layout()
plt.show()

### Pie Chart

plt.style.use("fivethirtyeight")

slices = [20, 30, 30, 20]
labels = ['Attendance', 'Midterm', 'Final', 'Assignments']
explode = [0, 0, 0.1, 0]

plt.pie(slices, labels=labels, explode=explode, shadow=True,
        startangle=90, autopct='%1.1f%%',
        wedgeprops={'edgecolor': 'black'})

plt.title("Grading Policy")
plt.tight_layout()
plt.show()

### Stacked Plot

plt.style.use("fivethirtyeight")


minutes = [1, 2, 3, 4, 5, 6, 7, 8, 9]

player1 = [1, 2, 3, 3, 4, 4, 4, 4, 5]
player2 = [1, 1, 1, 1, 2, 2, 2, 3, 4]
player3 = [1, 1, 1, 2, 2, 2, 3, 3, 3]

labels = ['player1', 'player2', 'player3']
colors = ['#6d904f', '#fc4f30', '#008fd5']

plt.stackplot(minutes, player1, player2, player3, labels=labels, colors=colors)

plt.legend(loc='upper left')

plt.title("Stacked Plot")
plt.tight_layout()
plt.show()

### Histogram

import random
import numpy as np

# grades = [random.randint(0,100) for i in range(1000)]

grades = np.random.normal(85, 13, 10000)
#bins = [50, 60, 70, 80, 90,100]

plt.hist(grades, bins= 50,edgecolor='black')

reference_line = np.mean(grades)
color = '#fc4f30'

plt.axvline(reference_line, color=color, label='Mean Score', linewidth=2)

plt.legend()

plt.title('Histogram')
plt.xlabel('Student Grades')
plt.ylabel('個數')

plt.tight_layout()

plt.show()

### Scatter Plot


plt.figure(figsize=(7,5), dpi=300)
plt.scatter(df['y4'], df['y5'], c=df['y6'], s=df['y6']*100,cmap='summer',
            edgecolor='black', linewidth=1, alpha=0.75)

cbar = plt.colorbar()
cbar.set_label('Y6數值大小')

# plt.xscale('log')
# plt.yscale('log')

plt.title('3D Scatterplot')
plt.xlabel('Y1')
plt.ylabel('Y2')

plt.ylim((0,13))

plt.show()


### Complex Graphs


df

# create a color palette
palette = plt.get_cmap('Set1')
 
# multiple line plot
num=0
#plt.figure(figsize=(5,3), dpi=150)
for column in df.drop(['x','y10'], axis=1):
    num+=1
    plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
 
plt.legend(loc=2, ncol=2)
 
# Add titles
plt.title("Line Plot With Several Values", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Time")
plt.ylabel("Score")

import matplotlib.pyplot as plt
plt.style.use('ggplot')

df_columns = df.drop(['x','y1'], axis=1).columns
palette = plt.get_cmap('tab20')

num=0

fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(8, 6), dpi=100)

for row in range(3):
    for col in range(3):
        ax[row,col].plot(df['x'],df[df_columns[num]],color=palette(num), linewidth=1.9, alpha=0.9, label=df_columns[num])
        ax[row,col].set_title(df_columns[num],loc='left', fontsize=14,color=palette(num))
        num+=1
fig.suptitle("Facet Grids", fontsize=14, fontweight=0, color='black', style='italic', y=1.02)
fig.text(0.5, 0.0, 'Common X', ha='center', fontsize=14)
fig.text(0.0, 0.5, 'Common Y', va='center', rotation='vertical', fontsize=14)

## Seaborn Module

![](../images/seaborn-func.png)

### Two Types of Functions

- Figure-level functions (Generic)
- Axex-level functions (Specific)

## Change the DPI

import seaborn as sns
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_context('notebook')
sns.set_style("ticks")

sns.set(style='darkgrid')

penguins = sns.load_dataset('penguins')
penguins.head()

# histogram
print(sns.__version__) # seaborn>=0.11.0
sns.displot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")


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

### Long-format vs. Wide-format Data

flights = sns.load_dataset("flights")
flights.head()

sns.relplot(data=flights, x="year", y="passengers", hue="month", kind="line")


sns.relplot(data=flights, x="month", y="passengers", hue="year", kind="line")

flights_wide = flights.pivot(index="year", columns="month", values="passengers")
flights_wide.head()


print(type(flights_wide))

sns.catplot(data=flights_wide, kind="box")

## Chinese Fonts Issues

- Find system-compatible Chinese fonts using the terminal command:

```
!fc-list :lang=zh
```

- Define the font to be used as well as the font properties in Python:

```
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
# rcParams['axes.unicode_minus']=False
myfont = FontProperties(fname='/Library/Fonts/Songti.ttc',
 size=15)
plt.title('圖表標題', fontproperties=myfont)
plt.ylabel('Y軸標題', fontproperties=myfont)
plt.legend(('分類一', '分類二', '分類三'), loc='best', prop=myfont)
```

- For a permanent solution, please read references.
    - Modify the setting file in matplotlib: `matplotlib.matplotlib_fname()` to get the file path
    - It's similar to: `/Users/YOUR_NAME/opt/anaconda3/lib/python3.7/site-packages/matplotlib/mpl-data/matplotlibrc`
    - Two important parameters: `font.family` and `font.serif`
    - Add the font name under `font.serif`. My case: `Source Han Sans`
    

## One can set the font preference permanently
## in the setting file
import matplotlib
matplotlib.matplotlib_fname()

from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
# rcParams['axes.unicode_minus']=False
#/Users/alvinchen/Library/Fonts/SourceHanSans.ttc
#'/System/Library/Fonts/PingFang.ttc'
def getChineseFont(size=15):  
    return FontProperties(fname='/Users/Alvin/Library/Fonts/SourceHanSans.ttc',size=size)  

print(getChineseFont().get_name())
plt.title('圖表標題', fontproperties=getChineseFont(20))
plt.ylabel('Y軸標題', fontproperties=getChineseFont(12))
plt.legend(('分類一', '分類二', '分類三'), loc='best', prop=getChineseFont())

## Permanent Setting Version
plt.rcParams['font.sans-serif']=["PingFang HK"]
plt.rcParams['axes.unicode_minus']= False

plt.plot((2,4,6), (3,5,7))
plt.title("中文標題")
plt.ylabel("y軸標題")
plt.xlabel("x軸標題")
plt.show()

## Seaborn
sns.set(font=['san-serif'])
sns.set_style("whitegrid",{"font.sans-serif":["PingFang HK"]})
cities_counter = [('好棒', 285), ('給我', 225), ('不要', 163), ('細柔', 136), ('吃飯', 130), ('小小', 124), ('深圳', 88), ('溫州', 67), ('小知', 66), ('大之', 45)]
sns.set_color_codes("pastel")
sns.barplot(x=[k for k, _ in cities_counter[:10]], y=[v for _, v in cities_counter[:10]])

## References

- [Python Graph Gallery](https://python-graph-gallery.com/)
- [Corey Schafer's YouTube matplotlib tutorial](https://youtu.be/UO98lJQ3QGI)
- [Matplotlib colormaps](https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html)
- [解決Python 3 Matplotlib與Seaborn視覺化套件中文顯示問題](https://medium.com/marketingdatascience/%E8%A7%A3%E6%B1%BApython-3-matplotlib%E8%88%87seaborn%E8%A6%96%E8%A6%BA%E5%8C%96%E5%A5%97%E4%BB%B6%E4%B8%AD%E6%96%87%E9%A1%AF%E7%A4%BA%E5%95%8F%E9%A1%8C-f7b3773a889b)
- [Day 11 : Google Colab 實用奧步篇 ( 連結硬碟、繪圖中文顯示問題 )](https://ithelp.ithome.com.tw/articles/10234373?sc=hot)
- [Souce Han Sans (open-source)](https://github.com/adobe-fonts/source-han-sans)
- [Seaborn Documentation](https://seaborn.pydata.org/tutorial/function_overview.html)

## Requirements

# %run ./get_modules.py
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