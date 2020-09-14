# Magic R

- This is a notebook based on an R kernel.
- To install the system default R kernel to work with Jupyter Notebook:
    - Open the terminal
    - Run the following commands in the terminal (running in RStudio does not work)
        ```
        # intiate R
        $ R
        # install packages
        install.package("IRkernel")
        IRkernel::installspec()
        ```


## Running R codes in Notebook

- After installing the R kernel, we can create an entire notebook, which is based on the system default R kernel.
- This notebook is an example. (awesome!)

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
filter(Sepal.Length > 5)