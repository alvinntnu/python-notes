# Input and Output


## Displaying Outputs

```
print(..., end = '\n')
print(..., sep = '***')
print('...' % (x, x, x))
```

## Getting User Input

```
input_text = input('Enter your name:')
```

## Reading Files

- Open a file 

```
fname = 'myfile.txt'
fp = open(fname, 'r')
fp = open(fname, 'w')
fp = open(fname, 'a')

fp.close()
```

- Read a text file (the entire contents of the file as one string

```
fp = open(fname, 'r')
text = fp.read()
lines = text.split('\n')
fp.close()
```

- Read a text file line by line

```
fp = open(fname, 'r')
lines = fp.readlines()
fp.close()
```

- Read lines and remove line breaks at the same time

```
lines = line.strip() for line in fp.readlines()
```

- Process the file line by line

```
fp = open(fname, 'r')
for eachLine in fp:
    # prceoss each line in turn
    
    print(eachLine, end=' ')
 fp.close()
```


with open('temp.txt', 'w') as f:
    f.write('hello world!\n' + 'This is my first sentence.' )
    
with open('temp.txt', 'r') as f:
    print([l for l in f.readlines()])
    
!rm temp.txt

## Writing Files

- prefered way:
```
with open('FILENAME', 'w') as f:
    f.write('hello world!')
```

fp = open('temp.txt', 'w')
while True:
    text = input('Enter text (end with blank):')
    if len(text)==0:
        break
    else:
        fp.write(text+'\n')
fp.close()

fp2 = open('temp.txt','r')
print([line for line in fp2])

:::{admonition,note}
To access files at specific positions:
```
file.seek()
file.read()
file.tell()
```

:::

## File/Directory Operation

```
import os
os.unlink()
os.rename()
os.chdir()
os.listdir()
os.getwd()
os.mkdir()
os.rmdir
os.path.exists()
os.path.isfile()
os.path.isdir()
```

## Command-Line Arguments

```
import sys

sys.argv 

```