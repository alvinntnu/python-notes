# Numpy

- Array structure
- Array subsetting
- Pass-by-reference vs. explicit copy
- Numpy broadcasting
- Masked Array


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

## Load from File

```
filedata = np.genformat('', delimiter=',')
filedata = filedata.astype('int32')
print(filedata
```
