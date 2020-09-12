# Program Structure

- A quick overview of control sturctures in Python

## If- structure

- Format

```
if test:
    DoThis()
elseif test:
    DoThat()
else:
    pass
```

## Types of Test

```
# comparisons
a > b
a == b
a != b

# membership
a in b
a not in b

# len check
len(x) > 5

# boolean value
fileopen
not fileopen

# Validation
x.isalpha()
x.isdigit()

# Calculation
(a*b) > 100.0
## use braces to force the calculation

# Combining Tests
and, or, not
```

## For Statement


names = ['Julia','Jane','Tom','Dickson','Harry']
tnum = range(1,5)
for p in names:
    for tn in tnum:
        print('%s has passed %f test' % (p, tn))
    

import itertools

for (p, tn) in zip(names, tnum):
    print('%s has passed %.0f test(s)' % (p, tn))

## While Statement

n=5
while n > 0:
    print(n)
    n-=1

n=5

while not n < 0:
    print(n)
    n-=1

## Break Statement



def doCommand(s):
    print('Hay %s! You are in the system now!!' % s)

while True:
    command=input('Enter your name:')
    if command=='exit':
        break
    else:
        doCommand(command)
print('bye')

while True:
    command=input('Enter your name:')
    if len(command)==0:
        continue
    elif command=='exit':
        print('Goodbye!')
        break
    else:
        doCommand(command)
print('bye')

## List Comprehensions

- A *list comprehension* is a way of dynamically creating a list elements in an elegant shorthand.

```
[expr for element in iterable if condition]
```

squares = [i**2 for i in range(1,10)]
print(squares)

squares2 = [i**2 for i in range(1,10) if not i % 2]
print(squares2)

text = 'This is a sample sentence long string.'

print([i.upper() for i in text if i.find('a')])

## Functions

```
def FUNCTION_NAME:
    PROCDURES
    
    reutrn RETURN_VALUES
```