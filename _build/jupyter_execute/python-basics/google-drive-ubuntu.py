# Google Drive with Ubuntu

So far Google has not released an official API for Linux with Google Drive.

This note illustrates how to mount Google Drive on Ubuntu using `ocamlfuse`.


## Install Ocamlfuse

```
$ sudo add-apt-repository ppa:alessandro-strada/ppa
$ sudo apt-get update
$ sudo apt-get install google-drive-ocamlfuse
```

## Authorize Google Account on Ubuntu

```
google-drive-ocamlfuse
```

Just follow the GUI instructions to allow Ubuntu to access the Google Drive.

## Create a Mounting Point in Local Machine

```
$ mkdir ~/google-drive
$ google-drive-ocamlfuse ~/google-drive
```

## Mount and Unmount Drive

```
fusermount -u ~/google-drive
```

## Benefits

- To mount the Google Drive using ocamlfuse, it is easier to access the files and directories in the terminal.
- Plus, all filenames and directory names show up properly!

## References

- [Mount google drive in linux using google-drive-ocamlfuse-client](https://www.tecmint.com/mount-google-drive-in-linux-using-google-drive-ocamlfuse-client/3/)