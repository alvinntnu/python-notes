#!/usr/bin/env python
# coding: utf-8

# # Web Crawler (Dcard)

# - Based on a [blog post](https://tlyu0419.github.io/2019/04/06/Crawl-Dcard/)

# In[1]:


# 載入使用的套件
import pandas as pd
import requests

import re


# In[2]:


# 檢視資料結構
ID = '235442609'
url = 'https://www.dcard.tw/_api/posts/' + ID
# 透過request套件抓下這個網址的資料
requ = requests.get(url)
# 初步檢視抓到的資料結構
requ.json()


# In[3]:


# 將抓下來的資料轉為DataFrame
ID = '235442609'
url = url = 'https://www.dcard.tw/_api/posts/' + ID
requ = requests.get(url)
rejs = requ.json()
pd.DataFrame(
    data=
    [{'ID':rejs['id'],
      'title':rejs['title'],
      'content':rejs['content'],
      'excerpt':rejs['excerpt'],
      'createdAt':rejs['createdAt'],
      'updatedAt':rejs['updatedAt'],
      'commentCount':rejs['commentCount'],
      'forumName':rejs['forumName'],
      'forumAlias':rejs['forumAlias'],
      'gender':rejs['gender'],
      'likeCount':rejs['likeCount'],
      'reactions':rejs['reactions'],
      'topics':rejs['topics']}],
    columns=['ID','title','content','excerpt','createdAt','updatedAt','commentCount','forumName','forumAlias','gender','likeCount','reactions','topics'])


# In[4]:


# 撰寫簡單的函數，透過輸入文章ID，就輸出文章的資料
def Crawl(ID):
    link = 'https://www.dcard.tw/_api/posts/' + str(ID)
    requ = requests.get(link)
    rejs = requ.json()
    return(pd.DataFrame(
        data=
        [{'ID':rejs['id'],
          'title':rejs['title'],
          'content':rejs['content'],
          'excerpt':rejs['excerpt'],
          'createdAt':rejs['createdAt'],
          'updatedAt':rejs['updatedAt'],
          'commentCount':rejs['commentCount'],
          'forumName':rejs['forumName'],
          'forumAlias':rejs['forumAlias'],
          'gender':rejs['gender'],
          'likeCount':rejs['likeCount'],
          'reactions':rejs['reactions'],
          'topics':rejs['topics']}],
        columns=['ID','title','content','excerpt','createdAt','updatedAt','commentCount','forumName','forumAlias','gender','likeCount','reactions','topics']))


# In[5]:


# 嘗試使用撰寫出的函數，抓取編號231030181的文章
Crawl(235442609)


# In[8]:


import random, time
# 一次讀取100篇最熱門的文章
url = 'https://www.dcard.tw/_api/posts?popular=true&limit=100'
resq = requests.get(url)
rejs = resq.json()
df = pd.DataFrame()
for i in range(len(rejs)):
    wait_time = random.randint(3,10)
    time.sleep(wait_time)
    df = df.append(Crawl(rejs[i]['id']),ignore_index=True)
print(df.shape)
df
# 將資料存到桌面
df.to_excel('dcard.xlsx')


# In[ ]:


# # 透過迴圈讀取10*100篇文章，若需讀取更多資料，可以將range(10)中的數值提升
# for j in range(2):
#     last = str(int(df.tail(1).ID)) # 找出爬出資料的最後一筆ID
#     url = 'https://www.dcard.tw/_api/posts?popular=true&limit=100&before=' + last
#     resq = requests.get(url)
#     rejs = resq.json()
#     for i in range(len(rejs)):
#         wait_time = random.randomint(3,10)
#         time.sleep(wait_time)
#         df = df.append(Crawl(rejs[i]['id']), ignore_index=True)
# print(df.shape)
# df


# In[ ]:


# 將資料存到桌面
# df.to_excel('dcard.xlsx')

