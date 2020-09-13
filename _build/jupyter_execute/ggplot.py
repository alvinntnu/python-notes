# ggplot


import numpy as np
import pandas as pd
from plotnine import *
from plotnine.data import mpg
%matplotlib inline

type(mpg)

pd.DataFrame.head(mpg)

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