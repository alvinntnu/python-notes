# Jupyer Notebook

This section includes notes relate to Jupyter Notebook (Book).


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

The following code runs in the terminal.

```
$ pip install jupyterthemes
$ jt -l
$ jt -t monokai
```

## Use a specific environment in notebook

- First, create a new environment
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