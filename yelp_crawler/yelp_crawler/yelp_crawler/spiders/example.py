# -*- coding: utf-8 -*-
import scrapy


class ExampleSpider(scrapy.Spider):
    name = "example"
    allowed_domains = ["http://www.yelp.com/"]
    start_urls = (
        'http://www.yelp.com/search?find_desc=&find_loc=nyc&ns=1#find_desc=chanel',
    )

    def parse(self, response):
        pass
