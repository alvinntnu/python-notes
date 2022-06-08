#!/usr/bin/env python
# coding: utf-8

# # Data Structure
# 
# - A quick overview of important data structures in Python
# 

# ## General Functions

# In[1]:


## Check object types
type(23)
type('some texts')
c = [1, 2, 'some text']
type(c)


# ## Basic Structures

# In[2]:


## Factory functions
int(4.0)
str(4)
list()
tuple('international')
dict(one=1, two=2)


# In[3]:


## Operations

## Modulus
15 % 4
## Exponentiation
4 ** 3
-4 ** 2 # minus applies to the result
(-4)**2

## Random numbers
import random
random.randint(1,10)


# ## `str` Object

# In[4]:


## Sequences

## Membership
'a' in 'track'
9 in [1, 2, 3, 4]
9 not in [1, 2, 3, 4]

## Concatenation
'this' + 'is'
' '.join(['this','is'])

## Subsetting Sequences
mylist = ['a', 'b','c','a word','e']
mylist[1]
mylist[0]
mylist[-1]
mylist[:3]
mylist[3][2:6]

##Strings
mystr = '   This is a seentence sample.  '
mystr.capitalize()
mystr.title()
mystr.upper()
mystr.lower()
mystr.rstrip()
mystr.lstrip()
mystr.strip()
mystr.find('is')
mystr.replace('is','was')

## String Comparison
## sapce > num > upper > lower
' ' > '0'
'0' > 'a'
'z' > 'm'
'm' > 'M'

## Special Characters
print('\O')
print('\t')
print('\n')
print('\'')
print('\"')
print('\\')

### Triple Quotes

multiline = """This is the first sentence
This is a second.
And A third. """

print(multiline)

### Format Strings
##`formatstring % (arguments to format)`
ntoys = 4
myname = 'Alvin'
length = 1234.5678
'%s has %d types' % (myname, ntoys)
'The toy is %.3f meters long' % (length)


# ## List
# 
# - A List is typically a sequence of objects all having the same type, of arbitrary length
# - It is a mutable data structure.
# - You can always append elements into the list, and it will automatically expand its size.
# - A List can include elements of different object types.
# 

# In[2]:


## Lists
list_empty = []
list_strings = ['Amy', 'Emma','Jane']
list_mixed = ['Amy','Emma', 5, 6]
list_embeded = [1, 3, 4, [99, 100, 101]]
len(list_empty)
len(list_mixed)
len(list_embeded)

## List operations
list_empty.append('Tom')
list_empty.append('John')
print(list_empty)
print(list_empty[1])
del list_empty[1]
print(list_empty)
list_mixed.index('Amy')
# list_mixed.index('Alvin')

## Other functions
## max(), min(), sum(), sorted(), reversed()


# - Python lists are zero-indexed (i.e., the index of the first element of the list is **0**.
# - Negative indices mean counting elements from the back.
# - Syntax for a slice of a list:
#     - `x[-2:]`: print the last two elements of `x`
#     - `x[2:]`: print all elements starting from the third element
#     - `x[start:end:step]`: the end is not included in the result

# In[8]:


x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
odd = x[::2]
even = x[1::2]
print(odd)
print(even)


# ## Tuples
# - A Tuple is typically a collection of objects of different types, of fixed length
# - Immutable (The tuple is a read-only data structure. It cannot be modified after it is created).

# In[6]:


## Tuples

tuple_numbers = (1,2,3,4,5,6,7)
tuple_strings = ('mon','tue','wed','thu','fri','sat','sun')
tuple_mixed = (1, 'mon', ['feb', 2])
print(tuple_mixed)
len(tuple_mixed)


# In[16]:


## unpacking with tuples
def powers(n):
    return n, n**2, n**3
x = powers(2)
print(x)

a,b,c = powers(2)
print(a, b, c)


# ## Dictionary
# 
# - Square brackets `[]` for list and curly brackets `{}` for dict.
# - A dict is for key-value mapping, and the key must be **hashable**.
# 
# 
# :::{note}
# 
# **From Official Python Documentation**
# 
# An object is hashable if it has a hash value which never changes during its lifetime, and can be compared to other objects. Hashable objects which compare equal must have the same hash value.
# 
# Hashability makes an object usable as a dictionary key and a set member, because these data structures use the hash value internally.
# 
# All of Python’s immutable built-in objects are hashable, while no mutable containers (such as lists or dictionaries) are. Objects which are instances of user-defined classes are hashable by default; they all compare unequal, and their hash value is their id().
# :::

# In[7]:


## Dictionary

dict_days = {'M': 'Monday', 'T':'Tuesday', 'W':'Wednesday'}
dict_days['M']
#dict_days['S']
dict_days['S']='Sunday'
dict_days['S']
'A' in dict_days
dict_days.keys()
dict_days.values()
dict_days.get('A','NA')


# In[13]:


wordfreq = {
    "the":100,
    "name": 10,
    "banana": 50
}

print(wordfreq["the"])

w = "hay"

if w in wordfreq:
    print(w)
    print("Its freq is ", wordfreq[w])
else:
    print("It is not observed in the corpus")
    
## set default values   
print(wordfreq.get(w, 0))

## Use keys or values

list(wordfreq.keys())
list(wordfreq.values())

## items()
list(wordfreq.items())

## combine two lists

wordfreq1 = {
    "hay":20
}

newwordfreq = dict(list(wordfreq.items())+ list(wordfreq1.items()))
print(newwordfreq)


# ## String Formatting

# In[8]:


## Format strings
print('Hello, {}! This is your No.{} book!'.format('Alvin',5))
print('An {0} a day keeps the {1} away.'.format('apple','doctor'))
print('An {1} a day keeps the {0} away.'.format('apple','doctor'))
print('The {noun} is {adj}!'.format(noun='book',adj='difficult'))

## Format strings with Dictionary
table = {'John': 98, 'Mary': 30, 'Jessica': 78, 'Goerge': 89, 'Jack': 45}
print('Jack: {0[Jack]:d}'.format(table))
print('Jack: {Jack:d}Jessica: {Jessica:d}'.format(**table))


# In[9]:


# wrapping strings
import textwrap
sentence= '''
美國大選首場總統辯論今晚登場，辯論會上總統川普頻頻插話，並與對手拜登互相人身攻擊，兩人鬥嘴不斷。美國有線電視新聞網（CNN）主持人直呼，這是史上最混亂總統辯論。
總統辯論一向被視為美國大選最重要環節之一，不少選民專心聆聽候選人政見，並為他們颱風及口條打分數。不過，今晚在俄亥俄州克里夫蘭市（Cleveland）登場的首場總統辯論，恐怕讓許多民眾直搖頭。
90分鐘辯論開始沒多久，總統川普與民主黨總統候選人拜登（Joe Biden）就吵個不停。川普頻頻插話並對拜登展開人身攻擊，不只酸拜登造勢活動只有兩三隻小貓，並指他一點都不聰明；拜登則多次面露不耐要川普「閉嘴」，並稱他是個「小丑」（clown）。'''

print(textwrap.fill(sentence, 20))


# In[10]:


## Old string formatting
import math
print('The value of pi is %1.2f' % math.pi) # specify number of digits before and after .


# ## List Comprehension
# 
# - A classic Pythonic way to create a list on the fly.

# In[19]:


mul3 = [n for n in range(1,101) if n%3 == 0]
mul3


# In[20]:


table = [[m*n for n in range(1,11)] for m in range(1,11)]
for row in table:
    print(row)


# ## Enumerate and Zip
# 
# - `enumerate()`: This is a handy function for loop-structure. We can get the loop index and the object from the looped structure at the same time. The result of `enumerate()` produces a tuple of the counter (default starts with zero) and the element of the list.
# - `zip()`: It takes elements from different lists and put them side by side.

# In[21]:


x = ["alpha", "beta", "gamma", "delta"]
for n,string in enumerate(x):
    print("{}: {}".format(n, string))


# In[22]:


x = ["blue", "red", "green", "yellow"]
y = ["cheese", "apple", "pea", "mustard"]
for a, b in zip(x, y):
    print("{} {}".format(a, b))


# ## Map, Filter, and Reduce
# 
# 
# - `map()`: to transform elements of a list using some function.
# - `filter()`: to short list the elements based on certain criteria.
# - `reduce()`: It scans the elements from a list and combines them using a function.
# 

# In[34]:


from functools import reduce

def maximum(a,b):
    if a > b:
        return a
    else:
        return b
 
x = [-3, 10, 2, 5, -6, 12, 0, 1]
max_x = reduce(maximum, x)
print(max_x)



## use reduce to 
## sum all positive words from a list
def concat_num(a,b):
    
    def pos(i):
        return i > 0
    
    out = filter(pos, [a,b])
    return(sum(out))

reduce(concat_num, x)


# ## Requirements

# In[11]:


# %load get_modules.py
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

