import requests
from bs4 import BeautifulSoup

BASE_URL = "https://en.wikipedia.org/wiki/Category:Companies_that_have_filed_for_Chapter_7_bankruptcy"

html_code = requests.get(BASE_URL, timeout=(6,27)) # connect, read
clean_html = BeautifulSoup(html_code.text, "html.parser")

selected_content = clean_html.find('div',{'class': 'mw-category'})
url_list_raw = selected_content.find_all("a", href=True)
url_list_text = [a.text for a in url_list_raw]

print(url_list_text)
