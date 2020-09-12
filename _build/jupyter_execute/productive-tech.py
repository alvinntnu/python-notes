# Phythonic Productivity Techniques

### Getting Help

import datetime
dir(datetime)

# Check objects with specific keywords
[_ for _ in dir(datetime) if 'date' in _.lower()]

### Virtual Environments

The following codes are run under terminal.

The following codes show how to create a virtual environment in the terminal.

```
which pip

pytho3 -m venv `DIRECTORY`
ls `DIRECTORY`

source activate `DIRECTORY`
pip list
pip install `PACKAGE_NAME`
source deactivate

```

### Change Notebook Themes

The following code runs in the terminal.

```
pip install jupyterthemes
jt -l
jt -t monokai
```

### Use a specific environment in notebook

- First, create a new environment
- Activate the new envrionment

```
source activate ENV_NAME
```

- install `ipykernal`

```
pip install --user ipykernel
```

- Add the environment kernal to Jupyter

```
python -m ipykernel install --user --name=myenv
```

- Check current environment kernels in Jupyter

```
jupyter kernelspec list
```

- Remove an environment from Jupyter kernel list

```
jupyter kernelspec uninstall myenv
```