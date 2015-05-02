# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html


# class TutorialItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()

from scrapy.item import Item, Field

# class DmozItem(Item):
#     title = Field()
#     link = Field()
#     desc = Field()

    # pass

class DmozItem(Item):
    title = Field()
    link = Field()
    desc = Field()

class YelpItem(Item):
    name = Field()
    address = Field()
    rate = Field()
    price_range = Field()
    category = Field()
    url = Field()


