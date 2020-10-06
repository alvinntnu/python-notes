# Numpy


## Why Numpy Array?

- If compared to built-in data structures (e.g., `list`), numpy array is more efficient, faster in computation.


## Basics
import numpy as np
## memory size and data type
x = np.array([1,2,3], dtype=np.float32)
print(x)
print(x.itemsize) # four bytes for item
print(x.nbytes)


print(x.ndim) ## get num of dimensions
print(x.shape) ## shape
print(x.dtype) ## data type
print(x.size) ## num of elements

%%time
## unary function
np.sin(x)

%%time
## math equivalent
## we have to use list comprehension
from math import sin
[sin(i) for i in x]

## Multidimensional array
## np supports at maximum 32-dimension array
x = np.array([range(10),range(10)])
print(x)
print(x.shape)

## Array Properties

- `arr.size`: Return number of elements in the `arr`
- `arr.shape`: Return dimensions of array (rows, columns)
- `arr.dtype`: Return type of elements in `arr`
- `arr.astype(dtype)`: Convert `arr` elements to type `dtype`
- `arr.tolist()`: Convert `arr` into a list


## Subsetting and Slicing

- `arr[START:END:STEP]`: Slicing elements
- `arr[4,4]`: Indexing specific element by (row, column)


## Subsetting
print(x[:,0]) # first column
print(x[:,1]) # second column
print(x[0,:]) # first row

## Subsetting sections of array
print(x[:, 1:]) # columsn from first to the last
print(x[:, ::2]) # all rows, every other columns
print(x[:, ::-1]) # all rows, reversed columns
print(x[:, 5:9:2]) # [, StartIndex:EndIndex:StepSize]
print(x[::-1,:]) ## all columns, reversed rows

## Subsetting 3D Array
## Principle: From outside in!
x = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(x)
print(x[0,1,1]) # should be 4
y = np.copy(x)
y[:,1,:]=[99,99]
print(y)

## Pass-by-reference
x = np.ones((2,3))
y = x
print(x)
print(y)
x[1,1]=2

## Note that both x and y objects are altered
print(x)
print(y)

## Creating Arrays

- `np.array([1,2,3])`: 1-D array
- `np.array([1,2,3],[4,5,6])`: 2-D array
- `np.zeros()`
- `np.ones((3,4))`: 3x4 aray with all values 1
- `np.eye(5)`: 5x5 array of 0 with 1 on diagonal (identity matrix)
- `np.linespace(0, 100, 6)`: Array of 6 evenly divided values from 0 to 100
- `np.arrange(0, 10, 3)`: Array of values from 0 to less than 10 with step 3
- `np.full((2,3), 8)`: 2x3 array with all values 8
- `np.random.ran(6,7)*100`: 6x7 array of random floats between 0-100
- `np.random.randint(5, size=(2,3))`: 2x3 array with random ints between 0-1

```{note}
In Python, the indices (esp. the closing indices) are often NOT inclusive.
```

## Initialize different types of Arrays

print(np.zeros((2,3)))
print(np.ones((2,3)))
print(np.full((2,3),99)) # create an array with self-defined default
x = np.array([[1,2,3],[4,5,6]])
print(x)
print(np.full_like(x,99)) # copy an array with default values

print(np.random.rand(4,3)) # random decimal numbers
print(np.random.randint(-10,10, size=(3,3))) ## random integer values
print(np.identity(5))
x1 = np.array([[1,2,3]])
x2 = np.array([1,2,3])
print(np.repeat(x1, 4, axis=0))
print(np.repeat(x2, 4, axis=0))
print(x1.shape)
print(x2.shape)

:::{admonition,important}
If the indexing object for the array is a non-tuple sequence object, another Numpy array (of type integer or boolean), or a tuple with at least one sequence object or Numpy array, then indexing creates copies.
:::

## 
x = np.ones((2,3))
y = x[:,[0,1,2]]
print(x)
print(y)
x[1,1] = 99
## Note that only x object is altered. y is stil the original!!!
print(x)
print(y)

## To explicity create a copy of an array
x = np.ones((2,3))
y = x.copy()
print(x)
print(y)
x[1,1]=99
print(x)
print(y)

## Numpy Broadcasting
X, Y = np.meshgrid(np.arange(2), np.arange(2))
print(X)
print(Y)
X + Y


x = np.array([0,1])
y = np.array([0,1])
print(x+y)
print(x + y[:,np.newaxis]) # the np.newaxis (None) makes copies of y along the dimension


## Adding/Removing Elements

- `np.append(arr, values)`
- `np.insert(arr, 2 values)`: Insert `values` into `arr` before index 2
- `np.delete(arr, 3, axis=0)`: Delete row (`axis=0`) on index 3 of `arr`
- `np.delete(arr, 3, axis=1)`: Delete column (`axis=1`) on index 3 of `arr`
- `np.repeat()`

np.repeat(3, 4)
np.repeat([2,8],[2,5])

## Concatenating/Slitting Arrays

- `np.concatenate((arr1, arr2), axis=0)`: Row-bind arrays
- `np.concatenate((arr1, arr2), axis=1)`: Column-bind arrays
- `np.split(arr, 3)`: Split `arr` into 3 sub-arrays based on rows
- `np.hsplit(arr, 3)`: Split `arr` into 3 euqal-sized sub-arrays based on the columns

x = np.random.randint(0,100,size=(3,4))
print(x)
print(np.split(x,3))
print(np.hsplit(x,2))

## Masked Array

## Masked Array
from numpy import ma
x = np.arange(10)
y = ma.masked_array(x , x<5) # copy=False
print(y)
print(y.shape)
x[6]=99
print(x)
print(y)
## The above shows that masked_array does not force an implicit copy operation

## Linear Algebra

- `np.add(arr, 2)`
- `np.substract(arr, 2)`
- `np.multiply(arr, 2)`
- `np.divide(arr, 2)`
- `np.power(arr, 2)`
- `np.array_equal(arr1, arr2)`
- `np.sqrt()`
- `np.sin()`
- `np.log()`
- `np.abs()`
- `np.ceil()`: Round up to the nearest int
- `np.floor()`
- `np.round()`

## Linear Algebra

## Matrix Multiplication
a = np.ones((2,3))
print(a)
b = np.full((3,2),2)
print(b)
print(np.matmul(a,b))

## Find the determinant
x = np.identity(3)
np.linalg.det(x)

:::{admonition,note}
Reference docs (https://docs.scipy.org/doc/numpy/reference/routines.linalg.html)
- Determinant
- Trace
- Singular Vector Decomposition
- Eigenvalues
- Matrix Norm
- Inverse
-  Etc...
:::

## Statistics

- `np.mean(arr)`
- `arr.sum()`
- `arr.max()`
- `arr.max(axis=0)`: Return max values of the rows
- `arr.max(axis=1)`: Return max values of the columns
- `arr.var()`:
- `arr.std()`
- `arr.correcoef()`: Returns correlation coefficient of array
- `np.where(arr==2)`: Return the index of which elements in `arr` is equal to 2
- `np.argmin(arr)`: Return the index of the min value of `arr`
- `np.argmax(arr)`: Return the index of the max value of `arr`

## Statistics
x = np.random.randint(0,100, size=(2,3))
print(x)
print(np.min(x, axis=0)) # min of each column
print(np.min(x, axis=1)) # min of each row
## 2D-array, first axis is the column?
print(np.sum(x, axis=0)) # sum of columsn

## Reorganizing Arrays
x = np.array([range(4),range(4)])
print(x)
y = x.reshape((4,2))
print(y)

# Stacking arrays
x = np.full((3,),3)
y = np.full((3,),6)
print(x)
print(y)
print(np.vstack([x,y]))
print(np.hstack([x,y]))

- Find which element has a specific value

## Search Elements in array
x = [1,2,3,4,0,1,2,3,4,11] 
x=np.array(x)
np.where(x == 2)

- Identify the first index of the element that is of the specific value

np.min(np.where(x==2))

- Find the index of the MIN/MAX

np.argmin(x)
np.argmax(x)

## Load from File

```
filedata = np.genformat('', delimiter=',')
filedata = filedata.astype('int32')
print(filedata
```


## Requirements

# %load ../get_modules.py
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

## References