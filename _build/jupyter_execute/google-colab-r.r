# Google Colab R

- Google is now supporting a colab notebook with the R engine
- We can run R codes directly on Colab
- [R Google Colab](https://colab.fan/r)
    - Colab with R kernel
    - With base-R installed
    - Run R codes immediately
    - No need to setup
    - Save a copy in your Drive


sessionInfo()

library(ggplot2)
library(dplyr)

head(iris)

iris %>%
ggplot(aes(Sepal.Length, Sepal.Width, color=Species))+
geom_point(alpha=0.7)