# Jupyer Notebook

This section includes notes relate to `Jupyter Notebook` as well as `jupyter-book`.

Recommended Readings:
- [Jupyter Notebook Tips and Tricks](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)
- [Notebook to slides](https://medium.com/pyradise/jupyter-notebook-tricks-slideshows-a057a39c0a23)



## Keyboard Shortcuts

- `Cmd + Shift + P`: Look up key shortcuts
- `Esc`: Get into command mode
- `Enter`: Get into edit mode
- While in command mode:
    - `A`: insert a new cell above
    - `B`: insert a new cell below
    - `DD`: delete the current cell
- `Shift + Tab`: Documentation (Docstring) of the object
- `Ctrl + Shift + -`: Split cells
- `Esc + F`: Find and replace
- `Esc + O`: Toggle cell outputs
- `Shift + UP/DOWN`: select multiple cells

- `Shift + M`: Merge multiple cells
- `Y`: Change cell to code
- `M`: Change cell to markdown

## Citation

- In-text citation

An example of in-text citation {cite}`deyne2016`.

- Bibliography

:::{admonition}

- To include the bibliography at the end:

```{bibliography} book.bib
:filter: docname in docnames
:style: unsrt
```

:::

```{bibliography} book.bib
:filter: docname in docnames
:style: unsrt
```



## Build the book

- Open `terminal` and set to the root of the book directory
- Build the book
```
$ jupyter-book build .
```

- Push to GitHub
```
$ git add .
$ git commit -m "XXX"
$ git push origin master
```

- Update GitHub Pages html files
```
$ git-ghp -n -p -f _build/html
```

## Publish Jupyter-book on Github

- Documentations
    - [Official documentation](https://jupyterbook.org/publish/gh-pages.html)
    - [Adding existing projects to Github repo](https://docs.github.com/en/github/importing-your-projects-to-github/adding-an-existing-project-to-github-using-the-command-line)
- Important Steps
    - Create an empty repo on Github
    - Open Terminal.
    - Change the current working directory to your local project.
    - Initialize the local directory as a Git repository: `$ git init`
    - Add commmits:
    ```
    $ git add .
    # To unstage a file, use 'git reset HEAD YOUR-FILE'.
    $ git commit -m "First commit"
    ```

    - Add the Github remote URL to the project local directory 
    ```
    $ git remote add origin remote repository URL
    # Sets the new remote
    $ git remote -v
    # Verifies the new remote URL
    ```

    - Push changes
    ```
    $ git push -u origin master
    # Pushes the changes in your local repository up to the remote repository you specified as the origin
    ```


    

##  Publish the book as Github Pages

- Install `ghp-import`
```
$ pip install ghp-import
```

- Create a branch of the repo called gh-pages and pushes to GitHub
```
$ ghp-import -n -p -f _build/html
```

```{note}
The `-n` refers to "not Jekyll"
```

## Change Notebook Themes

- The following code runs in the terminal.
```
$ pip install jupyterthemes
$ jt -l
$ jt -t monokai
```

- List of theme names
    - onedork
    - grade3
    - oceans16
    - chesterish
    - monokai
    - solarizedl
    - solarizedd
- Reset to the original default theme:
```
$ jt -r
```

## Use a specific environment in notebook

- First, create a new environment
```
! conda create --name ENV_NAME python=3.6
```
- Activate the new envrionment
```
$ source activate ENV_NAME
```

- install `ipykernal`
```
$ pip install --user ipykernel
```

- Add the environment kernal to Jupyter
```
$ python -m ipykernel install --user --name=myenv
```

- Check current environment kernels in Jupyter
```
$ jupyter kernelspec list
```

- Remove an environment from Jupyter kernel list
```
$ jupyter kernelspec uninstall myenv
```

```{note}
:class: dropdown

Sometimes, the pre-existing conda environment does not work properly. An easy solution is to create the environment again after you set up the jupyter notebook.

```


## IPython Magic Commands

- Magic commands from IPythhon
- Useful magic commands:
    ```
    # list all env variables
    %evn
    # set env variable
    %env OMP_NUM_THREADS=4
    ```
- Run external python codes/files
    ```
    %run ./hello-world.py
    ```
- Insert code from an external script
    ```
    %load SCRIPT_NAME
    ```
- Time the process
    ```
    ## Time the single run of the code in the cell
    %%time
    ## Run a statement 100,000 times by default and provide the mean of the fastest three times
    %timeit
    ## Shoe how much time your program spent in each function
    %prun STATEMENT_NAME
    ```
- Write files
    ```
    %%writefile FILENAME
    ## save contents of the cell to an external file
    %pycat
    ## show the syntax highlighted contents of an external file
    ```
- Debugging `%pdb`
- Render high-resolution graphs for Retina screens
    ```
    %config InlineBackend.figure_format = 'retina'
    ```
- Run shell commands
    ```
    !ls *.ipynb
    ```
- LaTex formula automatic rendering in markdown

    $P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$


## Hide or Remove Content

- See [documentation page](https://jupyterbook.org/interactive/hiding.html)

## Running R  and Python in the Same Notebook

- To do this, first we need to install relevant R packages to make the system default R kernel avaiable to the notebook
    ```
    # in the terminal
    $ R
    # in R
    install.package("IRkernel")
    IRkernel::installspec()
    ```
- Then install the python module `pip install rpy2`
- To use R and Python at the same time, the magic commend
    ```
    %load_ext rpy2.ipython
    %R library(ggplot2)
    ```
- Mac users may run into issues when installing `rpy2`. Please see [this solution](https://blog.csdn.net/u010555997/article/details/104078809). General principles:
    - Install Homebrew
    - Install ggc with `brew install gcc`
    - Install rpy2 using the updated gcc `env CC=/usr/local/Cellar/gcc/10.2.0/bin/gcc-10 pip install rpy2`

%run hello-world.ipynb

%%time
import time
for _ in range(1000):
    time.sleep(0.01)

import numpy
%timeit numpy.random.normal(size=100)

# List all Magic commands
%lsmagic

!ls *.ipynb

## Run R code chunks in notebook with python codes
%load_ext rpy2.ipython

%%R 
library(dplyr)
library(ggplot2)
data.frame(x = seq(1,100), y = seq(seq(1,100))) %>%
ggplot(aes(x,y)) + geom_point()