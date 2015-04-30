# find_best_malls

## Project Description

To determine what store should be selected to fit an empty slot in a (target) shopping mall so that the shopping mall realizes maximum profit from this decision. We have a dataset of all the malls in the USA, and all the stores of various categories within each mall. Each store falls into two groups: category (restaurant, appeals-to-parents, clothing, etc) and how high-ended each store in a category is (Armani, Louis Vuitton are high-ended stores in apparels, we can consider fast-food joints as low-ended stores and fine-dining is high-ended and so on). For example, a mall in a fancy neighbourhood in Manhattan might want to fill an empty slot with a Michael Kors outlet, while a mall in a modest neighbourhood in New Mexico might make more profit by renting out the space to a Tex-Mex food chain instead. The store that needs to fill in the empty slot will depend on various factors:
* The location of the mall and the presence of stores in nearby malls. This is under the assumption that the nearby malls will have the market in the area more-or-less accurately mapped and figured out. Thus, we can rely on the exisiting stores in the county where our target mall is located, to a large extent, to be able to predict that there is a good chance one of the existing stores should fill in the empty slot. (Data Source: Prof. Lawrence & Prof. Shasha's earlier research data)
* Demographic information of the area, such as earning capacity of the people in the county, the racial groups in the county and so forth, that is a key indicator of the degree of affordability, lifestyle and preferences of the people in the county where the target mall is located. (Data Source: BLS)
* The category and high-endedness of every brand in every mall in the dataset. (Data Source: Considering scraping Yelp for this information)

## technologies

* big data technologies: Pig and Spark 
* machine learning technologies 
    * using item-based collaborative filtering, 
    * user-based collaborative filtering
    * top-k items
    * content based recommendation 
    * occf
    * combine all the results together

## clusers
* use aws
* AWS run hadoop :http://aws.amazon.com/cn/elasticmapreduce/
* AWS run pig: http://aws.amazon.com/cn/elasticmapreduce/
* run spark on EC2: https://spark.apache.org/docs/latest/ec2-scripts.html

## grateful
We are grateful to Professor Shasha, Professor Lawrence and Professor McIntosh for their guidance.