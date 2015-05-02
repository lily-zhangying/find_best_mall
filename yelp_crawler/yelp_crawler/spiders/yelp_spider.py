from scrapy.spider import Spider
from scrapy.selector import Selector
from scrapy.log import log
from yelp_crawler.items import YelpItem
import csv
import urllib

class YelpSpider(Spider):
    name = "yelp"
    allowed_domain = ["yelp.com"]

    def get_start_urls():
        start_urls = []
        dir = "/Users/lily/workspace/find_best_mall/yelp_crawler/yelp_crawler/yelp_crawler/dataset/"
        with open(dir + 'sort_store_id_file.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            title_row = next(reader)
            for row in reader:
                mall = row[0]
# http://www.yelp.com/search?find_loc=USA&ns=1?#find_desc=prada
                url = "http://www.yelp.com/search?find_loc=USA&ns=1&find_desc=" + mall
                print url
                start_urls.append(url)
                # mall_list = row[7]
                # state = row[1]
                # for mall in mall_list.split("|"):
                #     mall = mall.strip()
                #     if ((mall in url_list) == False):
                #         url_list[mall] = state
                #         start_urls.append("http://www.yelp.com/search?find_loc=" + state + "&ns=1&find_desc=" + mall)
        csvfile.close()
        return start_urls


    start_urls = get_start_urls()

    # # test data
    # start_urls = [
    # "http://www.yelp.com/search?ns=1&find_desc=chanel",
    # 	"http://www.yelp.com/search?ns=1&find_desc=prada",
    # 	"http://www.yelp.com/search?ns=1&find_desc=enzo",
    # 	"http://www.yelp.com/search?ns=1&find_desc=hm"
    # ]

    def parse(self, response):
        sel = Selector(response)
        stores = sel.xpath('//div[@data-key="1"]')
        items = []
        for first_store in stores:
            item = YelpItem()
            url = urllib.unquote(response.url).decode('utf8')
            item['name'] = url[url.index("find_desc") + 10: len(url)]
            # item['name'] = first_store.xpath('.//a[@class="biz-name"]/span/text()').extract()
            # item['address'] = first_store.xpath('.//address/text()').extract()
            #@todo todo just left 3.5 use pace to split the string
            item['rate'] = first_store.xpath('.//div[@class="rating-large"]/i/@title').extract()
            #@todo replace $$$$ with numbers
            item['price_range'] = first_store.xpath('.//div[@class="price-category"]/span/span/text()').extract()
            item['category'] = first_store.xpath('.//span[@class="category-str-list"]/a/text()').extract()
            # item['url'] = first_store.xpath('cd.//a[@class="biz-name"]/@href').extract()
            items.append(item)
        return items
