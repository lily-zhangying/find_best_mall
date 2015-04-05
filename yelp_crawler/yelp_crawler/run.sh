#!/bin/bash

# run sprider
NOW=$(date +"%m-%d-%Y-%T-")
log_file=$NOW"yelp_spider.log"
csv_file="./"$NOW"yelp_malls.csv"

/usr/local/bin/scrapy crawl yelp --output=$csv_file -t csv -s LOG_FILE=$log_file