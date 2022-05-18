# Data Visualization 2

- This unit covers the layered framework for data visualization, i.e., the Grammar of Graphics.
- The codes implement the ideas of layered graph building using the module, `plotnine`, which is essentially the Python version of R ggplot2.

import numpy as np
import pandas as pd
from plotnine import *
from plotnine.data import mpg, mtcars
%matplotlib inline

type(mpg)

## Data Frame Preparation

pd.DataFrame.head(mpg)

pd.DataFrame.head(mtcars)

## Basic Graphs

(ggplot(mpg) +
aes(x = 'class', fill = 'class') +
geom_bar(size=20))

(
    ggplot(mpg) +
    aes(x = 'class', y = 'hwy', fill = 'class') +
    geom_boxplot() +
    labs(x = 'Car Classes', y = 'Highway Milage')
)

(
    ggplot(mpg) +
    aes(x = 'cty', y = 'hwy', fill = 'class') +
    geom_point(alpha = .7)
)

(
ggplot(mpg) +
aes(x = 'cyl', y = 'hwy', fill = 'class') +
    geom_boxplot()
)

## More Complex Graphs

- Visualize four-dimensions (4-D)

(
    ggplot(mtcars, aes('wt','mpg', color='factor(gear)')) +
    geom_point() +
    facet_wrap('~cyl')+
    theme_bw()+
    labs(x='Weight', y='Miles/gallon')+
    scale_color_discrete(name='Forward Gear Number (Factor)')
)

(
    ggplot(mtcars, aes('wt','mpg',color='factor(gear)'))+
    geom_point()+
    geom_smooth(method="lm")+
    theme_bw()
)

- Visualize 5-D 
    - x
    - y
    - facet
    - color
    - size

(
    ggplot(mtcars, aes('wt','mpg',color='factor(gear)', size='cyl'))+
    geom_point()+
    facet_wrap('~am')+
    theme_bw()

)

- Visualize 6-D
    - x
    - y
    - facet (2 dimensions)
    - color
    - size

(
    ggplot(mtcars, aes('wt', 'mpg', color='factor(gear)', size='cyl'))+
    geom_point()+
    facet_grid('am~carb')+
    theme_bw()
)

## More?

- Check [Hans Rosling's famous visualization of global population](https://www.ted.com/speakers/hans_rosling?language=en) (a dynamic graphing)