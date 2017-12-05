
# coding: utf-8

# In[1]:


from pyspark import SparkConf, SparkContext
import pandas as pd



conf = SparkConf().setMaster("local[*]").setAppName("MovieRecommender")
sc = SparkContext(conf=conf)



sharlock = sc.textFile("input/sherlock.txt")




# # Most frequent words



words = sharlock.flatMap(lambda line: line.split())




word_tuple = words.map(lambda word: (word, 1))


results = word_tuple.reduceByKey(lambda v1,v2: (v1+v2)).collect()

print(results)

result_pd = pd.DataFrame(results)
result_pd.to_csv("aaa2.csv")
