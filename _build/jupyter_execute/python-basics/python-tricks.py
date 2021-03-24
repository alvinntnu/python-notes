# Python Tricks

- This notebook mostly includes useful Python tricks provided by [Dan Bader](https://dbader.org/).
- He has a very useful book: [Python Tricks: A Buffet of Awesome Python Features](https://www.tenlong.com.tw/products/9781775093305). Highly recommended!!
- Also, I include other tricks that came from different sources as well as many trial-and-errors. Not very organized, but always good to keep all these referneces.

## Module Path

## Find module path
import re, os
#print(re.__path__[0]) # if the module is a directory
print(re.__file__)
path = os.path.abspath(re.__file__)

## Sorting Iterables

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

## Find Modules in Memory

#!pip list
import sys
modulenames = set(sys.modules) & set(globals())
allmodules = [sys.modules[name] for name in modulenames]
print(allmodules)

import sys
import pprint

# pretty print loaded modules
pprint.pprint(sys.modules)

## Format Dict Output

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

## Unpacking Tuples

## Unpacking function arguments

def myfunc(x, y, z):
    print(x, y, z)

tuple_vec = (1, 0, 1)
dict_vec = {'x': 1, 'y': 0, 'z': 1}
dict_vec2 = {'z':3, 'x': 1, 'y': 2}
myfunc(*tuple_vec) # pass all necessary function arguments all at once, using a tuple
myfunc(**dict_vec) # pass all necessary function arguments all at once, using a dict
myfunc(**dict_vec2) # dict would help match which values go with which function argument!

 ## `filter()`

- `filter(function, iterable)` is useful when you need to filter elements in a sequence by using self-defined methods (which returns either True or False.
- If function is `None`, then the `fiter()` returns only `True` elements from the `iterable`.
- [Source](https://www.programiz.com/python-programming/methods/built-in/filter)

# list of letters
letters = ['a', 'b', 'd', 'e', 'i', 'j', 'o']

# function that filters vowels
def filterVowels(letter):
    vowels = ['a', 'e', 'i', 'o', 'u']

    if(letter in vowels):
        return True
    else:
        return False

filteredVowels = filter(filterVowels, letters)

print('The filtered vowels are:')
for vowel in filteredVowels:
    print(vowel)

# random list
randomList = [1, 'a', 0, False, True, '0']

filteredList = filter(None, randomList)

print('The filtered elements are:')
for element in filteredList:
    print(element)

## Asterisk `*`

That `FUNCTION(*iterable)` is passing all of the items in the `iterable` into the `function` call as separate arguments, without us even needing to know how many arguments are in the list.

fruits = ['lemon', 'pear', 'watermelon', 'tomato']
print(fruits)
print(fruits[0], fruits[1], fruits[2], fruits[3])
print(*fruits)

def transpose_list(list_of_lists):
    return [
        list(row)
        for row in zip(*list_of_lists)
    ]

transpose_list([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
[[1, 2, 3], [4, 5, 6], [7, 8, 9]]


## Find Index of Specific Values in List

Four methods to identify indices of List element which are specific values.

# Python3 code to demonstrate  
# finding indices of values 
# using naive method  
  
# initializing list  
test_list = [1, 3, 4, 3, 6, 7] 
  
# using naive method 
# to find indices for 3 
res_list = [] 
for i in range(0, len(test_list)) : 
    if test_list[i] == 3 : 
        res_list.append(i) 
        
res_list

# Python3 code to demonstrate  
# finding indices of values 
# using list comprehension  
  
# initializing list  
test_list = [1, 3, 4, 3, 6, 7] 
  
# printing initial list  
print ("Original list : " + str(test_list)) 
  
# using list comprehension 
# to find indices for 3 
res_list = [i for i in range(len(test_list)) if test_list[i] == 3] 
          
# printing resultant list  
print ("New indices list : " + str(res_list)) 

# Python3 code to demonstrate 
# finding indices of values 
# using enumerate() 

# initializing list 
test_list = [1, 3, 4, 3, 6, 7] 

# printing initial list 
print ("Original list : " + str(test_list)) 

# using enumerate() 
# to find indices for 3 
res_list = [i for i, value in enumerate(test_list) if value == 3] 
		
# printing resultant list 
print ("New indices list : " + str(res_list)) 


# Python3 code to demonstrate 
# finding indices of values 
# using filter() 

# initializing list 
test_list = [1, 3, 4, 3, 6, 7] 

# printing initial list 
print ("Original list : " + str(test_list)) 

# using filter() 
# to find indices for 3 
res_list = list(filter(lambda x: test_list[x] == 3, range(len(test_list)))) 
		
# printing resultant list 
print ("New indices list : " + str(res_list)) 
