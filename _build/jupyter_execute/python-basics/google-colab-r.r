# Google Colab R

- Google is now supporting a Colab notebook with the R kernel
- We can run R codes directly on Colab
- R Colab Notebook Link: [R Google Colab](https://colab.fan/r)
    - Colab with R kernel
    - With base-R installed
    - Run R codes immediately
    - No need to setup
    - Save a copy in your Drive


## Demonstration

This notebook uses the R kernel.


sessionInfo()

library(ggplot2)
library(dplyr)

head(iris)

iris %>%
ggplot(aes(Sepal.Length, Sepal.Width, color=Species))+
geom_point(alpha=0.7)