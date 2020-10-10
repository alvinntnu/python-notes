# Jupyer Notebook

- This section includes notes on how to use `Jupyter Notebook` as well as `jupyter-book`.

- Recommended Readings:
    - [Jupyter Notebook Tips and Tricks](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)
    - [Notebook to slides](https://medium.com/pyradise/jupyter-notebook-tricks-slideshows-a057a39c0a23)
    - Check notebook extension documentations (Very useful) (See {ref}`nbextensions`)



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

An example of in-text citation 

```
{cite}`deyne2016`.
```


- Bibliography

To include the bibliography at the end of the file):

```
{bibliography} book.bib
:filter: docname in docnames
:style: unsrt
```


## Special Blocks in Jupyer Book


- [sphinx-book-theme documentation](https://sphinx-book-theme.readthedocs.io/en/latest/reference/demo.html#admonitions)
- `note`
- `warning`
- `caution`
- `danger`
- `error`
- `hint`
- `important`
- `tip`
- self-defined `admonition`

## Add Images/Figures

- [Documentation](https://jupyterbook.org/content/figures.html)

## Cross-reference and Citations

- Add labels before the bookmarked positions (e.g., sections):

```
(LABEL_NAME)=
```

- Cross-reference:

```
{ref}`LABEL_NAME`
```

## Build the book

- Open the `terminal` and set to the root of the book directory
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

- Create GitHub Pages html files as a branch
```
$ git-ghp -n -p -f _build/html
```

- Updates the github pages
    - build the jupyter book
    - update the repository by pushing all changes
    - update the github pages via
    ```
    $ ghp-import -n -p -f _/build/html
    ```

## Publish Jupyter-book on Github

- Documentations
    - [Official documentation](https://jupyterbook.org/publish/gh-pages.html)
    - [Adding existing projects to Github repo](https://docs.github.com/en/github/importing-your-projects-to-github/adding-an-existing-project-to-github-using-the-command-line)
- Important Steps
    - Create an empty repo on Github
    - Open Terminal.
    - Change the current working directory to your local project.
    - Initialize the local directory as a Git repository: 
    
    ```
    $ git init
    ```
    
    - Add commmits:
    
    ```
    $ git add .
    ```
    
    - To unstage a file, use 'git reset HEAD YOUR-FILE'.
    
    ```
    $ git commit -m "First commit"
    ```

    - Add the Github remote URL to the project local directory and verify it
    
    ```
    $ git remote add origin remote repository URL
    $ git remote -v
    ```
    
    - Push changes
    ```
    $ git push -u origin master
    ```


    

##  Publish the book as Github Pages

- Install `ghp-import`
```
$ pip install ghp-import
```

- Create a branch of the repo called gh-pages and pushes to GitHub
```
$ git-ghp -n -p -f _build/html
$ ghp-import -n -p -f _build/html
```

```{note}
The `-n` refers to "not Jekyll"
```

## Change Notebook Themes

- Install `jupyterthemes`

```
$ pip install jupyterthemes
```

- Change themes

To preserve the toolbars (`-T`), Logo (`-N`), and kernel logo (`-kl`)

```
$ jt -l
$ jt -t monokai (-T -N -kl)
```

- Restart the browser after setting the new theme
- List of available theme names
    - onedork
    - grade3
    - oceans16
    - chesterish
    - monokai
    - solarizedl
    - solarized
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
Sometimes, the pre-existing conda environment does not work properly. An easy solution is to create the environment again after you set up the jupyter notebook. It is recommended to always create a new virtual environment for a new project.
```

- Update `conda`:

```
!conda update -n base -c defaults conda
```


(nbextensions)=

## Install Notebook Extensions

- [nbextensions documentation](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html)

- There are in general three steps
    - Install the modudle `jupyter_contribe_nbextensions`
    - Install javascript and css files
    - Enable specific extensions
```
$ pip install jupyter_contrib_nbextensions
$ jupyter contrib nbextension install --user
$ jupyter nbextension enable <nbextension require path>
```

- Use `jupyter_extensions_configuaror` to manguage the extensions
- Recommended extensions:
    - `varInspector`: very useful to check objects in the current memory of the kernel.
    - `ScratchPad`: very good for debugging in notebook
    - `Table on Content (2)`: Good to view the outline structure of the notebook file.
    - `Spellchker`
    - `Live Markdown Preview`

## IPython Magic Commands

- Magic commands from IPython
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

- See [Jupyterbook Documentation](https://jupyterbook.org/interactive/hiding.html)

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

## Memory Issues

When seeing error messages as shown below:

```
IOPub data rate exceeded.
The notebook server will temporarily stop sending output
to the client in order to avoid crashing it.
```

Try:
    
```
jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10
```

## Requirements

# %load get_modules.py
import pkg_resources
import types
def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, 
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

        # Some packages are weird and have different
        # imported names vs. system/pip names. Unfortunately,
        # there is no systematic way to get pip names from
        # a package's imported name. You'll have to add
        # exceptions to this list manually!
        poorly_named_packages = {
            "PIL": "Pillow",
            "sklearn": "scikit-learn"
        }
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]

        yield name
        
        
imports = list(set(get_imports()))

# The only way I found to get the version of the root package
# from only the name of the package is to cross-check the names 
# of installed packages vs. imported packages
requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name!="pip":
        requirements.append((m.project_name, m.version))

for r in requirements:
    print("{}=={}".format(*r))