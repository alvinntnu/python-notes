# Google Colaboratory (Colab)

- Anyone with a google acount can use it.
- An on-line python platform to run codes
- Pre-installed several common packages for machine learning

## How to use?

- Get a Gmail account
- Upload/save data files on Google Drive
- Upload/save jupyter notebooks on Google Drive
- Now you are ready to go: [Google Colab](https://colab.research.google.com/)
    - Download the notebook files and save it on the Google Drive
    - Run the notebook on Colab
 
## How to install packages on Colab?

- You may install the Python package on your Google Colab terminal by runing the following code in the notebook:
```
!pip install PACKAGE_NAME
```

## How to access files/directories on your Google Drive?

- You need to mount your Google Drive first
```
from google.colab import drive
drive.mount('/content/gdrive')
##check directories of cwd
!ls -l '/content/gdrive/My Drive'
```

```{note}
When you mount the drive, you will be asked to log into your Gmail and then Google would give you the authorization text. Copy it and paste it back to your Colab notebook.
```

## Run .py script on Colab

```
! python3 '/content/gdrive/My Drive/SCRIPT_NAME.py'
```