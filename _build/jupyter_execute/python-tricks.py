# Python Tricks

- This notebook includes useful Python tricks provided by [Dan Bader](https://dbader.org/).
- All the tricks are cumulated via different sources as well as many trial-and-errors. Not very organized.

# convert a int List into str List
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

# How to sort a Python dict by value
# (== get a representation sorted by value)
xs = {'a': 4, 'b': 3, 'c': 2, 'd': 1}
sorted(xs.items(), key=lambda x: x[1])

import operator
sorted(xs.items(), key=operator.itemgetter(0))
#operator.itemgette


for item in xs.items():
    print(item)


# sort a list
from random import randrange
num = [randrange(1,100) for _ in range(20)]
#sorted(num, reverse=True)
#sorted(num) # default ascending order
print(num)
num.sort()# this mutates the list and does not return anything
print(num) # list is mutable


# sort a tuple?

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