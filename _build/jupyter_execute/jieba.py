# Chinese Word Segmentation (jieba)


import jieba
from jieba import posseg

# set dictionary

#jieba.set_dictionary()
#jieba.load_userdict()

text = '據《日經亞洲評論》網站報導，儘管美國總統川普發起了讓美國製造業回歸的貿易戰，但包括電動汽車製造商特斯拉在內的一些公司反而加大馬力在大陸進行生產。另據高盛近日發布的一份報告指出，半導體設備和材料以及醫療保健領域的大多數公司實際上正擴大在大陸的生產，許多美國製造業拒絕「退出中國」。'

print(' '.join(jieba.cut(text, cut_all=False, HMM=True))+'\n')
print(' '.join(jieba.cut(text, cut_all=False, HMM=False))+'\n')
print(' '.join(jieba.cut(text, cut_all=True, HMM=True))+'\n')

text_pos = posseg.cut(text)
#print(type(text_pos))
for word, tag in text_pos:
    print(word+'/'+tag)