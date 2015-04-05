from scrapy.spider import Spider
from scrapy.selector import Selector

from yelp_crawler.items import DmozItem

class DmozSpider(Spider):
   name = "dmoz"
   allowed_domains = ["dmoz.org"]
   start_urls = [
       "http://www.dmoz.org/Computers/Programming/Languages/Python/Books/",
       "http://www.dmoz.org/Computers/Programming/Languages/Python/Resources/"
   ]

   def parse(self, response):
       sel = Selector(response)
       sites = sel.xpath('//ul/li')
       items = []
       for site in sites:
           item = DmozItem()
           item['title'] = site.xpath('a/text()').extract()
           item['link'] = site.xpath('a/@href').extract()
           item['desc'] = site.xpath('text()').extract()
           items.append(item)
       return items