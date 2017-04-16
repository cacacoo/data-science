# coding=utf-8
import re
from bs4 import BeautifulSoup
import requests
from time import sleep
from collections import Counter
import matplotlib.pyplot as plt

html = requests.get("http://www.example.com").text
soup = BeautifulSoup(html, 'html.parser')

first_paragraph = soup.find('p')
first_paragraph_text = first_paragraph.text
first_paragraph_words = first_paragraph.text.split()


url = "http://shop.oreilly.com/category/browse-subjects/" + \
        "data.do?sortby=publicationData&page=1"
soup_for_oreilly = BeautifulSoup(requests.get(url).text, 'html.parser')

tds = soup_for_oreilly('td', 'thumbtext')
print len(tds)


def is_video(td):
    pricelabels = td('span', 'pricelabels')
    return (len(pricelabels) == 1 and
            pricelabels[0].text.strip().startswith("Video"))

# 필요없는 데이터 걸러내기, 비디오는 제낀다
print len([td for td in tds if not is_video(td)])


def book_info(td):
    title = td.find("div", "thumbheader").a.text
    author_name = td.find("div", "AuthorName").text
    authors = [x.strip() for x in re.sub("^By ", "", author_name).split(",")]
    isbn_link = td.find("div", "thumbheader").a.get("href")
    isbn = re.match("/product/(.*)\.do", isbn_link).groups()[0]
    date = td.find("span", "directorydate").text.strip()

    return {
        "title" : title,
        "authors": authors,
        "isbn": isbn,
        "date": date
    }


base_url = url = "http://shop.oreilly.com/category/browse-subjects/" + \
        "data.do?sortby=publicationData&page="
books = []

NUM_PAGES = 2

for page_num in range(1, NUM_PAGES + 1):
    print "souping page", page_num, ",", len(books), "found so far"
    url = base_url + str(page_num)
    soup = BeautifulSoup(requests.get(url).text, "html.parser")

    for td in soup('td', 'thumbtext'):
        if not is_video(td):
            books.append(book_info(td))

    sleep(30)


def get_year(book):
    """August 2004 이런식으로 저장하고 있어서 2014만 뗀다"""
    return int(book["date"].split()[1])


year_counts = Counter(get_year(book) for book in books
                      if get_year(book) <= 2017)
years = sorted(year_counts)
book_counts = [year_counts[year] for year in years]
plt.plot(years, book_counts)
plt.ylabel("# of data books")
plt.title("Data is Big!")
plt.show()
