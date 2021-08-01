from urllib.parse import urlencode

import numpy as np
from scrapy import Selector
import scrapy


class PureSpider(scrapy.Spider):
    name = 'purespider'
    start_url = 'https://pureportal.coventry.ac.uk/en/organisations/coventry-university/persons/'
    count = 1

    def start_requests(self):
        yield scrapy.Request(self.start_url,
                             callback=self.parse)

    def parse(self, response):
        self.logger.debug("started")
        sel = Selector(response)

        for res in sel.css('h3.title'):
            link = res.css('a::attr(href)').extract_first()
            baseAuthorLink = response.urljoin(link+'/publications/')


            yield scrapy.Request(baseAuthorLink, callback=self.parse_author)
        next_page = response.xpath("//a[@class='nextLink']/@href").extract_first()
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)

    def parse_author(self, response):
        sel = Selector(response)
        for res in sel.css('div.result-container > div.rendering_researchoutput  > h3.title > a::attr(href)').extract():
            paperUrl = res
            if not paperUrl or not len(paperUrl.strip()):
                yield None
            yield scrapy.Request(paperUrl, callback=self.parse_paper)

    def parse_paper(self, response):
        self.count = self.count + 1
        print('Crawled Pages: ' + str(self.count))
        sel = Selector(response)
        paperUrl = response.url
        title = sel.css('div.rendering > h1::text').extract_first()
        s = ""
        authorslist = sel.css('p.relations.persons ::text').extract()
        authors = s.join(authorslist);

        year = sel.css('tr.status > td > span.date ::text').extract_first()
        description = sel.css('div.textblock ::text').extract_first()
        fingerprintlist = sel.css('span.concept-wrapper > span.thesauri ::text').extract()
        s = ','
        fingerprint = s.join(list(set(fingerprintlist)))
        if (not title) or not len(title.strip()):
            paperUrl = response.url
        if (not description) or not len(description.strip()):
            description = ''
        item = {'title': title, 'PaperUrl': paperUrl, 'Authors': authors, 'PublishedDate': year,
                'description': description, 'fingerprint':fingerprint}
        print('Crawled Doc:')
        print(item)
        yield item