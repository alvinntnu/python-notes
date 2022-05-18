#!/usr/bin/env python
# coding: utf-8

# # Google Colaboratory (Colab)
# 
# - Anyone with a Google acount can use it
# - An on-line python platform to run codes
# - Pre-installed with several common packages for machine learning
# 
# ## How to use?
# 
# - Get a Gmail account
# - Upload/save data files on Google Drive
# - Upload/save jupyter notebooks (copies) on Google Drive
# - Now you are ready to go: [Google Colab](https://colab.research.google.com/)
#     - Download the notebook files and save it on your Google Drive
#     - Run the notebook on Colab
# - You can also open any python notebooks that are on the public GitHub repositories. (You can save a copy of it after you make changes to the notebook.)
#  
# ## How to install packages on Colab?
# 
# - You may install the Python package on your Google Colab terminal by runing the following code in the notebook:
# ```
# !pip install PACKAGE_NAME
# ```
# 
# ```{note}
# You may need to install the package every time you run the code on Colab.
# ```
# 
# ## How to access files/directories on your Google Drive?
# 
# - You need to mount your Google Drive first
# ```
# from google.colab import drive
# drive.mount('/content/gdrive')
# ##check directories of cwd
# !ls -l '/content/gdrive/My Drive'
# ```
# 
# ```{note}
# When you mount the drive, you will be asked to log into your Gmail and then Google would give you the authorization code. Copy it and paste it back to your Colab notebook.
# ```
# 
# ## Run .py script on Colab
# 
# ```
# ! python3 '/content/gdrive/My Drive/SCRIPT_NAME.py'
# ```
# 
# ## Free GPU Computing
# 
# - To enable GPU backend for your Colab notebook:
#     - *Runtime* -> *Change runtime type* -> *Hardware Accelerator*-> *GPU*
# - To cross-check whether the GPU is enabled, you can run the following cell in the colab notebook:
# ```
# import tensorflow as tf
# tf.test.is_gpu_available()
# tf.test.gpu_device_name()
# ```
# 
# or:
# 
# ```
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# ```
# 
# The expected output would be:
# 
# ```
# /device:GPU:0
# ```
# 
# ```{note}
# When uploading data files onto the Google Drive, remember to uncheck the box, which determines whether to automatically convert txt files to Google-compatible formats (e.g., txt -> gdoc). This is undesirable.
# ```
# 
# - Check the GPU spec
# 
# ```
# !nvidia-smi
# ```
