# Data Structure

- An quick overview of important data structures in Python


## Check object types
type(23)
type('some texts')
c = [1, 2, 'some text']
type(c)

## Factory functions
int(4.0)
str(4)
list()
tuple('international')
dict(one=1, two=2)

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

## Tuples

tuple_numbers = (1,2,3,4,5,6,7)
tuple_strings = ('mon','tue','wed','thu','fri','sat','sun')
tuple_mixed = (1, 'mon', ['feb', 2])
print(tuple_mixed)
len(tuple_mixed)

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