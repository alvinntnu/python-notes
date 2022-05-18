sessionInfo()

library(ggplot2)
library(dplyr)

head(iris)

iris %>%
ggplot(aes(Sepal.Length, Sepal.Width, color=Species))+
geom_point(alpha=0.7)
