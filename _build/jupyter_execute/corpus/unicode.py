# Unicode

- Dealing with unicode texts can be tedious sometimes.
- It is good to have a basic understanding of the [Unicode Character Database](https://unicodebook.readthedocs.io/index.html)
- In particular, this notebook focuses on the Python module `unicodedata`.

## Character Name

import unicodedata

print(unicodedata.name('A'))
print(unicodedata.name('我'))

## Characrer to Numbers

print(unicodedata.numeric('四'))  # any character
print(unicodedata.numeric('壹'))  # any character
#print(unicodedata.digit('四')) # digits only
#print(unicodedata.decimal('六'))

## Look-up By Name

print(unicodedata.lookup('CJK UNIFIED IDEOGRAPH-6211'))
print(unicodedata.lookup('LEFT CURLY BRACKET'))

## Unicode Category

print(unicodedata.category('a'))
print(unicodedata.category('A'))
print(unicodedata.category('{'))
print(unicodedata.category('。'))
print(unicodedata.category('$'))
print(unicodedata.category('我'))

## Normalization

- Ways of normalization: NFD, NFC, NFKD, NFKC
- Suggested use:**NFKC**
- Meaning:
    - D = Decomposition (will change the length of the original form)
    - C = Composition 
    - K = Compatibility (will change the original form)

## Chinese characters with full-width English letters and punctuations
text = '中英文abc,，。.．ＡＢＣ１２３'
print(unicodedata.normalize('NFKD', text))
print(unicodedata.normalize('NFKC', text))  # recommended method
print(unicodedata.normalize('NFC', text))
print(unicodedata.normalize('NFD', text))

text = 'English characters with full-wdiths ＡＢＣ。'

## Encode the string in ASCII and find compatible characters
print(
    unicodedata.normalize('NFKC',
                          text).encode('ascii',
                                       'ignore').decode('utf-8', 'ignore'))
print(
    unicodedata.normalize('NFKD',
                          text).encode('ascii',
                                       'ignore').decode('utf-8', 'ignore'))

## Encode the string in ASCII and but remove ASCII-incompatible chars

print(
    unicodedata.normalize('NFC',
                          text).encode('ascii',
                                       'ignore').decode('utf-8', 'ignore'))
print(
    unicodedata.normalize('NFD',
                          text).encode('ascii',
                                       'ignore').decode('utf-8', 'ignore'))

text = 'Klüft skräms inför på fédéral électoral große'

unicodedata.normalize('NFKD', text).encode('ascii',
                                           'ignore').decode('utf-8', 'ignore')

## Normalizing Texts

text = "中文ＣＨＩＮＥＳＥ。！＝=.= ＾o＾ 2020/5/20 alvin@gmal.cob@%&*"

# remove puncs/symbols
print(''.join(
    [c for c in text if unicodedata.category(c)[0] not in ["P", "S"]]))

# select letters
print(''.join([c for c in text if unicodedata.category(c)[0] in ["L"]]))

# remove alphabets
print(''.join(
    [c for c in text if unicodedata.category(c)[:2] not in ["Lu", 'Ll']]))

# select Chinese chars?
print(''.join([c for c in text if unicodedata.category(c)[:2] in ["Lo"]]))

```{note}
It seems that the unicode catetory **Lo** is good to identify Chinese characters?
```