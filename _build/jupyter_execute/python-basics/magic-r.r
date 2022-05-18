library(ggplot2)
library(tidyverse)
library(quanteda)

getwd()
version
sessionInfo()

head(iris)

ggplot(iris, aes(Species, Sepal.Length, fill=Species)) +
geom_boxplot(notch=T)

iris %>% 
filter(Sepal.Length > 5) %>% 
head(10)
