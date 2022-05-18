# Python Tricks

- This notebook mostly includes useful Python tricks provided by [Dan Bader](https://dbader.org/).
- He has a very useful book: [Python Tricks: A Buffet of Awesome Python Features](https://www.tenlong.com.tw/products/9781775093305). Highly recommended!!
- Also, I include other tricks that came from different sources as well as many trial-and-errors. Not very organized, but always good to keep all these referneces.

## Find module path
import re, os
#print(re.__path__[0]) # if the module is a directory
print(re.__file__)
path = os.path.abspath(re.__file__)

## convert a int List into str List
a = [1,5,10]
type(a)

b = str(a)
print(b)
print(type(b))

c = [str(i) for i in a]
print(c)
print(type(c))

# convert a int tuple into str tuple
e = [(2,3,4),(23,42,54)]
f = [tuple([str(i) for i in tp]) for tp in e]
print(e)
print(f)

## How to sort a Python dict by value
# (== get a representation sorted by value)
xs = {'a': 4, 'b': 3, 'c': 2, 'd': 1}
sorted(xs.items(), key=lambda x: x[1])

## How to sort a Python dict using operator
import operator
sorted(xs.items(), key=operator.itemgetter(0))
#operator.itemgette


for item in xs.items():
    print(item)


## sort a list
from random import randrange
num = [randrange(1,100) for _ in range(20)]
#sorted(num, reverse=True)
#sorted(num) # default ascending order
print(num)
num.sort()# this mutates the list and does not return anything
print(num) # list is mutable


## sort a tuple?

tuple_ex = (2,50, -2, 30)
#tuple_ex
tuple_ex_sorted = sorted(tuple_ex)
print(tuple_ex)
print(tuple_ex_sorted)
type(tuple_ex_sorted)



# into string 
def convertTuple(tup, split=''): 
    str =  split.join(tup) 
    return str
  
# Driver code 
print(convertTuple(tuple([str(i) for i in tuple_ex]), split=','))

tuple_lexicon = [
    ('the', 'det',[23, 800]),
    ('off', 'prep', [70, 100])
]

# sort according to alphabets?
#print(sorted(tuple_lexicon, key=lambda x:x[0]))

print(tuple_lexicon)
print(tuple_lexicon.sort())
print(tuple_lexicon)

#!pip list
import sys
modulenames = set(sys.modules) & set(globals())
allmodules = [sys.modules[name] for name in modulenames]
print(allmodules)

import sys
import pprint

# pretty print loaded modules
pprint.pprint(sys.modules)

## How to output a dictionary
my_mapping = {'a': 23, 'b': 42, 'c': 0xc0ffee}
print(my_mapping)
# The "json" module can do a much better job:
import json
import pprint
print(json.dumps(my_mapping, indent=4, sort_keys=True))


# # Note this only works with dicts containing
# # primitive types (check out the "pprint" module):
# json.dumps({all: 'yup'})


import pprint
pprint.pprint(my_mapping, depth=1)
mPP = pprint.PrettyPrinter()
mPP.pprint(my_mapping)


my_mapping

# from modulefinder import ModuleFinder
# finder = ModuleFinder()
# finder.run_script("SCRIPT_ONLY")
# for name, mod in finder.modules.items():
#     print(name)

## Unpacking function arguments

def myfunc(x, y, z):
    print(x, y, z)

tuple_vec = (1, 0, 1)
dict_vec = {'x': 1, 'y': 0, 'z': 1}
dict_vec2 = {'z':3, 'x': 1, 'y': 2}
myfunc(*tuple_vec) # pass all necessary function arguments all at once, using a tuple
myfunc(**dict_vec) # pass all necessary function arguments all at once, using a dict
myfunc(**dict_vec2) # dict would help match which values go with which function argument!