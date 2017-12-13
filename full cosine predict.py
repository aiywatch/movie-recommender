
# coding: utf-8

# In[55]:


from pyspark import SparkConf, SparkContext
import pandas as pd
import numpy as np
from math import sqrt
import time
# from sklearn.metrics import mean_squared_error


# In[2]:


sc = SparkContext.getOrCreate()
rating_data = sc.textFile("input/ml-100k/u.data")

rating_data_test, rating_data_train = rating_data.randomSplit(weights=[0.2, 0.8], seed=1)


# # Similarity score

# In[41]:


def _extract_user_rating(line):
    data = line.split('\t')
    return (int(data[0]), [(int(data[1]), float(data[2]))])

def rate_substract_mean(line):
    k = line[0]
    v = line[1]
    rating_list = v[0]
    acc_rating = v[1]
    count_rating = v[2]
    user_mean = acc_rating / count_rating
    
    return [(k, (m_id, rating-user_mean)) for m_id, rating in rating_list]

def _computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, numPairs)

def _filter_movies(line):
    movie1 = line[1][0]
    movie2 = line[1][1]

    return movie1 < movie2

def _makePairs(line):
    user_id = line[0]
    (movie1, rating1) = line[1][0]
    (movie2, rating2) = line[1][1]

    return ((movie1, movie2), (rating1, rating2))



# In[45]:


start_time = time.time()

user_rating_lists = rating_data_train     .map(_extract_user_rating)     .aggregateByKey(([], 0., 0.), lambda g1,v2: (g1[0]+v2, g1[1]+v2[0][1], g1[2]+1), 
                    lambda g1,g2: (g1[0]+g2[0], g1[1]+g2[1], g1[2]+g2[2])) \
    .flatMap(rate_substract_mean) 
    
join_lists = user_rating_lists.join(user_rating_lists)
moviePairSimilarities = join_lists.filter(_filter_movies)     .map(_makePairs)     .groupByKey()     .mapValues(_computeCosineSimilarity).persist()

# moviePairSimilarities.saveAsPickleFile("input/movie-sims-obj2")
elapsed_time = time.time() - start_time


# In[46]:


print(elapsed_time)
moviePairSimilarities.takeSample(False, 10)


# # Prediction part

# In[47]:


def load_sim_dict(from_file=False):
    if from_file:
        sim_movie = sc.pickleFile("input/movie-sims-obj2/")
        
    else:
        sim_movie = moviePairSimilarities
        
    sim_movie_local = sim_movie.collect()
    sim_dict = {}
    for sm in sim_movie_local:
        key = sm[0]
        value = sm[1]

        sim_dict[key] = value
    
    return sim_dict

def _extract_movie_data(line):
    data = line.split('|')
    return (int(data[0]), data[1])

def rate_movie(user_ratings, predicted_movie):
    
    numerator = 0
    denominator = 0
    
    normalize_rating = {1:-1, 2:-0.5, 3:0, 4:0.5, 5:1}

    for movie_id, rating in user_ratings:
        if(predicted_movie < movie_id):
            m1 = predicted_movie
            m2 = movie_id
        else:
            m2 = predicted_movie
            m1 = movie_id
        
        if (m1, m2) in sim_dict:
            sim_score, number_of_record = sim_dict[(m1, m2)]
        else:
            sim_score, number_of_record = (0,0)

        numerator += sim_score*normalize_rating[rating]
        denominator += sim_score
    
    predicted_rating = numerator/denominator if denominator else 0
    predicted_rating = 0.5*(predicted_rating+1)*4 + 1
    
    if predicted_rating > 5:
        predicted_rating = 5.0
    elif predicted_rating < 1:
        predicted_rating = 1.0
    
    return predicted_rating


# In[51]:


sim_dict = load_sim_dict()
movie_data = sc.textFile("input/ml-100k/u.item")
movie_dict = dict(movie_data.map(_extract_movie_data).collect())

user_lists = rating_data_train.map(_extract_user_rating).reduceByKey(lambda v1,v2: v1+v2).persist()
test_set = rating_data_test.map(_extract_user_rating).reduceByKey(lambda v1,v2: v1 + v2).collect()

predicted_ratings = []

for user_id, movie_rating in test_set:
    _, user_ratings = user_lists.filter(lambda line: line[0]==user_id).collect()[0]
    
    for m_id, rating in movie_rating:
        predicted_ratings += [(user_id, m_id,rate_movie(user_ratings, m_id), rating)]


# ## Result and accuracy

# In[58]:


predicted_ratings = pd.DataFrame(predicted_ratings)

rmse = sqrt((predicted_ratings[2]-predicted_ratings[3])**2)
print(rmse)

# rmse = sqrt(mean_squared_error(predicted_ratings[2],predicted_ratings[3]))
# predicted_ratings

