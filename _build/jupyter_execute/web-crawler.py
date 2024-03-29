# Web Crawler

import requests
from bs4 import BeautifulSoup as soup  # HTML data structure

# extract article hyperlinks from an index page
def extractArtLinks(url):
    r = requests.get(url, cookies={'over18':'1'})
    page_soup = soup(r.text, "html.parser")
    containers = page_soup.findAll("div", {"class": "r-ent"})
    art_links = []
    for container in containers:
        # Finds all link tags "a" from within the first div.
        art_link = container.find('a')
        if art_link:
            #print(art_link['href'])
            #print(container.find('div',{'class':'title'}).get_text())
            art_meta = container.find('div',{'class':'meta'})
            #print(art_meta.find('div',{'class':'author'}).get_text())
            #print(art_meta.find('div',{'class':'date'}).get_text())

            art_links.append({
                'push': container.find('div',{'class':'nrec'}).get_text(),
                'title': container.find('div',{'class':'title'}).get_text().strip(),
                'date': art_meta.find('div',{'class':'date'}).get_text(),
                'author': art_meta.find('div',{'class':'author'}).get_text(),
                'link': art_link['href'],
                'text': extractArtText('https://www.ptt.cc' + art_link['href'])
            })

    return(art_links)

# find the previous index page link
def findPrevIndex(url):
    r = requests.get(url, cookies={'over18':'1'})
    page_soup = soup(r.text,"html.parser")
    btn = page_soup.select('div.btn-group > a')
    up_page_href = btn[3]['href']
    next_page_url = 'https://www.ptt.cc' + up_page_href
    return(next_page_url)

# extract article contents from  the article hyperlink
def extractArtText(url):
    r = requests.get(url, cookies={'over18':'1'})
    page_soup = soup(r.text, "lxml")
    #print(page_soup.find("div",{"id":"main-content"}).get_text())
    art_text=page_soup.select('div#main-content', limit=1)[0].text
    return(art_text)

# main()
num_of_index_page = 2
board_name = 'Food'
url = 'https://www.ptt.cc/bbs/{}/index.html'.format(board_name)
all_links =[]
for page in range(1,num_of_index_page):
    all_links = all_links + extractArtLinks(url)
    url = findPrevIndex(url)
len(all_links)

type(all_links[2])
print(all_links[2])

print('Push: {push:s} \n'
      'title: {title:s} \n'
      'date: {date:s} \n'
      'author: {author:s} \n'
      'link: {link:s} \n'
      'text: {text:.5} \n'.format(**all_links[2]))



:::{admonition} Exercise
How to seperate post texts from push texts?
:::
