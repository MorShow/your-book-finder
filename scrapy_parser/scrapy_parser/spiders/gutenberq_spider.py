import time
import re
import scrapy

from scrapy import Request
from scrapy.spiders import Response

from typing import Optional


class GutenberqSpider(scrapy.Spider):
    name = 'gutenberq'
    pages_count = None

    def __init__(self, pages_ratio=0.1, **kwargs):
        self.pages_ratio = float(pages_ratio)
        self.start_urls = ['https://www.gutenberg.org/ebooks/1']
        super().__init__(**kwargs)

    def parse(self, response: Response) -> Request:
        pages_info = response.css('li.breadcrumb.next>a>span::text').get()
        pages_count = round(int(pages_info.replace(',', '').split()[0]) * self.pages_ratio)

        for i in range(1, pages_count + 1):
            yield Request(f'https://www.gutenberg.org/ebooks/{i}', callback=self.parse_book_info)

    def parse_book_info(self, response: Response) -> dict:
        if response.status == 200:
            page_title_author = response.css('h1#book_title::text').get()
            book_info = response.css('div.summary-text-container>span::text')[1].get()
            delimiter, delimiter_length = ' by ', len(' by ')  # text format: '<title> by <author>'
            delimiter_index = page_title_author.rfind(delimiter)
            if delimiter_index == -1:  # it is an unstable way so the code is mode complex that it could be
                delimiter2 = ' is ', len(' is ')
                delimiter_index, delimiter2_index = (book_info.find(delimiter),
                                                     book_info.find(delimiter2))
                title = book_info[:delimiter_index].replace('\"', '')
                author = book_info[delimiter_index + delimiter_length:delimiter2_index]
            else:
                title = page_title_author[:delimiter_index]
                author = page_title_author[delimiter_index+delimiter_length:]

            date = None
            rows_meta = response.css('table#about_book_table>tr>td::text')
            for row in rows_meta:
                row = row.get()
                pattern = re.compile(r'[A-Za-z]{3} [0-9]+, [0-9]{4}')  # Date format: 'Dec 19, 2000'
                if pattern.search(row):
                    date = time.strftime('%Y/%m/%d', time.strptime(row, '%b %d, %Y'))
                    break

            hrefs = response.css('table.files>tr>td.noscreen::text')
            for href in hrefs:
                href = href.get()  # selector object -> str
                pattern = re.compile('\.txt\.utf-8$')
                if pattern.search(href):  # we need only plain text (in utf-8) to form a dataset
                    yield Request(href,
                                  callback=self.parse_book_text,
                                  cb_kwargs={'title': title,
                                             'author': author,
                                             'date': date,
                                             'info': book_info})
        else:
            self.logger.warning(f'The page is missing: {response.url}')

    def parse_book_text(self,
                        response: Response,
                        title: str,
                        author: str,
                        date: Optional[str],
                        info: str) -> dict:
        if response.status == 200:
            book_content = response.text.strip().replace('\r', '')
            metadata_end_index = book_content.find('*** START OF THE PROJECT GUTENBERG EBOOK')
            text_end_index = book_content.find('*** END OF THE PROJECT GUTENBERG EBOOK')

            metadata_lines = book_content[:metadata_end_index].splitlines()

            language = next((line for line in metadata_lines if line.startswith('Language')), None)
            language = language.split(':')[1]
            book_text = book_content[book_content.find('\n', metadata_end_index):text_end_index].strip()

            yield {
                'title': title,
                'author': author,
                'date': date,
                'info': info.strip(),
                'language': language,
                'text': book_text
            }
        else:
            self.logger.warning(f'The page is missing: {response.url}')
