# Data Visualization I

![](../image/seaborn-func.png)
(source: [documentation](https://seaborn.pydata.org/tutorial/function_overview.html)

```{note}
Still like the vidualization tools in R better.
Seaborn is not intuitive, esp. for long-wide conversion of data frames.
```

## Matlibplot

- [Python Graph Gallery](https://python-graph-gallery.com/)

```{note}
We can increase the dpi of the matplotlib parameters to get image of higher resolution in notebook. Instructions are included below. And please note that the dpi setting has to GO before the magic inline command. (The magic inline commened resets the dpi to default.)
```

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

## Change DPI for higher resolution in notebook
%matplotlib inline
matplotlib.rcParams['figure.dpi']= 100

# Change DPI when saving graphs in files
# matplotlib.rc("savefig", dpi=dpi)


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

## Long-format vs. Wide-format Data

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

- [解決Python 3 Matplotlib與Seaborn視覺化套件中文顯示問題](https://medium.com/marketingdatascience/%E8%A7%A3%E6%B1%BApython-3-matplotlib%E8%88%87seaborn%E8%A6%96%E8%A6%BA%E5%8C%96%E5%A5%97%E4%BB%B6%E4%B8%AD%E6%96%87%E9%A1%AF%E7%A4%BA%E5%95%8F%E9%A1%8C-f7b3773a889b)
- [Day 11 : Google Colab 實用奧步篇 ( 連結硬碟、繪圖中文顯示問題 )](https://ithelp.ithome.com.tw/articles/10234373?sc=hot)
- [Souce Han Sans (open-source)](https://github.com/adobe-fonts/source-han-sans)