# Word Segmentation Issues



from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

# Set Parameter Path
MODEL_PATH = '/Users/Alvin/Dropbox/Corpus/CKIP_WordSeg/data/'
ws = WS(MODEL_PATH)
pos = POS(MODEL_PATH)
#ner = NER(MODEL_PATH)

## Raw text corpus 
sentence_list = ['他每天開車上班，整天塞啊！',
                 '他每天開著車，沒客人。',
                '他每天開車上班得開一小時，整天塞啊',
                '這間店每天八點準時開門。',
                '這小女孩很高呢！','這小女孩很高'
                '這小女孩好高！',
                '這小女孩很愛看書',
                '這本書好像是他寫的',
                '這本書像是他寫的',
                 '他慢慢地走進教室裡','他慢慢得走進教室裡',
                 '這本書看起來很有趣',
                 '他拿起這本書',
                 '這小男孩嚐看味道怎麼樣','這小男孩嚐看看味道怎麼樣', ## 屈折詞綴切分
                 '現代年輕人都吃得起高檔餐廳',
                 '這孩子吃到棒棒糖好開心','這孩子吃到掉滿地','這孩子吃到全身都是'
            ]

word_list = ws(sentence_list)
pos_list = pos(word_list)
def print_word_pos_sentence(word_sentence, pos_sentence):
    assert len(word_sentence) == len(pos_sentence)
    for word, pos in zip(word_sentence, pos_sentence):
        print(f"{word}({pos})", end="\u3000")
    print()
    return
    
for i, sentence in enumerate(sentence_list):
    print()
    print(f"'{sentence}'")
    print("=="*5)
    print_word_pos_sentence(word_list[i],  pos_list[i])
    print('\n')
    