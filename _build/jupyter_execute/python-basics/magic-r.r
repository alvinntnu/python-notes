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
    - This step is necessary if we want to use the default system R kernel instead of the R provided by Anaconda
 
- After installing the R packages, we can use the R kernel in a jupyter notebook. And the entire notebook has to use the same R kernel.
- If we need to combine python and R codes in one notebook, we need to do the following:
    - Install the rpy2 module
    ```
    pip install rpy2
    ```
    - Use magic command to switch to R codes
    ```
    %%R
    library(dplyr)
    
    %% R -i DUMP_PYTHON_OBJECTS_FOR_R
    ```
    
    - Some other parameters
    
    ```
    %%R -i df -w 5 -h 5 --units in -r 200
    # import df from global environment
    # make default figure size 5 by 5 inches with 200 dpi resolution

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
filter(Sepal.Length > 5) %>% 
head(10)