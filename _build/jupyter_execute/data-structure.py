# Data Structure

- An quick overview of important data structures in Python
- A List is typically a sequence of objects all having the same type, of arbitrary length. 
- A Tuple is typically a collection of objects of different types, of fixed length.


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

## Format strings
print('Hello, {}! This is your No.{} book!'.format('Alvin',5))
print('An {0} a day keeps the {1} away.'.format('apple','doctor'))
print('An {1} a day keeps the {0} away.'.format('apple','doctor'))
print('The {noun} is {adj}!'.format(noun='book',adj='difficult'))

## Format strings with Dictionary
table = {'John': 98, 'Mary': 30, 'Jessica': 78, 'Goerge': 89, 'Jack': 45}
print('Jack: {0[Jack]:d}'.format(table))
print('Jack: {Jack:d}Jessica: {Jessica:d}'.format(**table))

# wrapping strings
import textwrap
sentence= '''
美國大選首場總統辯論今晚登場，辯論會上總統川普頻頻插話，並與對手拜登互相人身攻擊，兩人鬥嘴不斷。美國有線電視新聞網（CNN）主持人直呼，這是史上最混亂總統辯論。
總統辯論一向被視為美國大選最重要環節之一，不少選民專心聆聽候選人政見，並為他們颱風及口條打分數。不過，今晚在俄亥俄州克里夫蘭市（Cleveland）登場的首場總統辯論，恐怕讓許多民眾直搖頭。
90分鐘辯論開始沒多久，總統川普與民主黨總統候選人拜登（Joe Biden）就吵個不停。川普頻頻插話並對拜登展開人身攻擊，不只酸拜登造勢活動只有兩三隻小貓，並指他一點都不聰明；拜登則多次面露不耐要川普「閉嘴」，並稱他是個「小丑」（clown）。'''

print(textwrap.fill(sentence, 20))

## Old string formatting
import math
print('The value of pi is %1.2f' % math.pi) # specify number of digits before and after .