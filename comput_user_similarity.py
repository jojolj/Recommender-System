# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 19:06:11 2023

@author: justd
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 11:31:18 2023
Recommender system Assignment 1 version 2
no filter more than 100 ratings

@author: justd
"""
#%%
# Data processing
import pandas as pd
import numpy as np
import scipy.stats

# Visualization
import seaborn as sns

# Similarity
from sklearn.metrics.pairwise import cosine_similarity
#%%
ratings = pd.read_csv('D:/MasterStudies/2023/Recommender_Systems/assignment/ml-latest-small/ratings.csv')
# Get the dataset information # There are 100836 ratings(rows), and no-null rating
ratings.info()
#%%
#%%%
# Read movie.csv file
movies = pd.read_csv('D:/MasterStudies/2023/Recommender_Systems/assignment/ml-latest-small/movies.csv')
# Merge ratings and movies datasets
df = pd.merge(ratings, movies, on='movieId', how='inner')
 
#%%%

#%% Create User-Movie Matrix
# Transform the dataset into a matrix. 
# The value of the matrix is the user rating of the movie if there is a rating. 
# Otherwise, it shows 'NaN'. There are many ‘NaN’ value in the matrix.
matrix = df.pivot_table(index='userId', columns='title', values='rating')
matrix.head()
#%% Data Normalization
# Normalize the rating by extracting the average rating of each user.
# After normalization, the movies with a rating < the user's average rating get 
# a negative value, and the movies with a rating > the user's average rating 
# get a positive value.
# Normalize user-item matrix
matrix_normaliz = matrix.subtract(matrix.mean(axis=1), axis = 'rows')
matrix_normaliz.head()
#%% Identify Similar Users
# User similarity matrix using Pearson correlation
user_similarity = matrix_normaliz.T.corr()
user_similarity.head()
#%%
# Above is the code for exercise a and b
#%%
#%% Below is the code for exercise c


#%% Prediction
# Get total numbers of users and movies
users_num = ratings['userId'].max()
movies_num = ratings['movieId'].max()
# Calculate the rating means of different users
rating_mean = []
for i in range(users_num):
    mean = ratings[ratings['userId']==i+1]['rating'].mean()
    rating_mean.append(mean)
    if i < 10:
        print("rating_mean", i, rating_mean[i-1])
# Definition of the predictions of userId to movieId inputed
def predictions(userId,movieId):
    # For given movieId, get all users who rate it.
    r_b = ratings[ratings['movieId']==movieId]['userId']
    # For given movieId, get all ratings of known users
    r_b_p = ratings[ratings['movieId']==movieId]['rating']
    # For all users who rate it, get their rating means
    r_b_mean = []
    for i in r_b:
        mean = rating_mean[i-1]
        r_b_mean = np.append(r_b_mean, mean)
    # Calculate the similarity between input userId and every
    # user who rates the input movieId times the difference between
    # the rate of user to input movieId and the rating mean of this
    # user, them sum them.
    for i in range(len(r_b)):
        sum = 0
        minus = r_b_p.iloc[i] - r_b_mean[i]
        if not pd.isnull(user_similarity[userId][r_b.iloc[i]]):
            sim = user_similarity[userId][r_b.iloc[i]]
        else:
            sim = 0
        times = sim * minus
        sum += times
    # Calculate the similarity between input userId and every 
    # user who rates the input movieId
    for i in range(len(r_b)):
        if not pd.isnull(user_similarity[userId][r_b.iloc[i]]):
            sim = user_similarity[userId][r_b.iloc[i]]
        else:
            sim = 0
        sum_sim = 0
        sum_sim += sim
    # If the sum of similarities is 0, it cannot be calculated
    if sum_sim ==0:
        return('The prediction cannot be calculated')
    else:
        # Calculate the prediction
        pre = rating_mean[userId - 1] + (sum / sum_sim)
        # If the output is greater than 5.0, reset it to 5.0
        if pre > 5.0:
            pre = 5.0
        return(pre)

    
#%%
# Below is the code for exercise for exercise d
#%%
#%%
# Below is the code for exercise for exercise d
#%%
#%% select user ID = 1
selected_userid = 1

# Remove picked user ID from the candidate list
user_similarity.drop(index=selected_userid, inplace=True)

# Take a look at the data
user_similarity.head()
#%% In the user similarity matrix, the values range from -1 to 1, 
# where -1 means opposite movie preference and 1 means same movie preference.

#n = 10 means we would like to pick the top 10 most similar users for user ID 1.

# set the user_similarity_threshold to be 0.3, 
# meaning that a user must have a Pearson correlation coefficient of at least 0.3 
# to be considered as a similar user.

# After setting the number of similar users and similarity threshold, 
# sort the user similarity value from the highest and lowest, 
# then printed out the most similar users' ID and the Pearson correlation value.
 
# Number of similar users
n = 10

# similarity threashold
similarity_threshold = 0.3

# Get top n similar users
similar_users = user_similarity[user_similarity[selected_userid]>similarity_threshold][selected_userid].sort_values(ascending=False)[:n]

# Print out top n similar users
print(f'The similar users for user {selected_userid} are', similar_users)
#%% Movies that the target user has rated
selected_userid_rated = matrix_normaliz[matrix_normaliz.index == selected_userid].dropna(axis=1, how='all')
selected_userid_rated
#%%
# Movies that similar users rated. Remove movies that none of the similar users have watched
similar_user_movies = matrix_normaliz[matrix_normaliz.index.isin(similar_users.index)].dropna(axis=1, how='all')
similar_user_movies
#%%%%%%%%%
# Remove the user ID 1 rated movies from the similar_user_movies list
similar_user_movies.drop(selected_userid_rated.columns,axis=1, inplace=True, errors='ignore')

similar_user_movies
#%%%
# Create a dictionary to store item_scores
movie_score = {}

# Loop through items
for i in similar_user_movies.columns:

  print("i:", i)
  #the ratings for movie i
  movie_rating = similar_user_movies[i]
  print("similar_user_movies[i]:", similar_user_movies[i])
  print("movie_rating:",movie_rating)
  # Create a variable to store the score
  total = 0
  # Create a variable to store the number of scores
  count = 0
  # Loop through similar users
  for a in similar_users.index:

    # If the movie has rating
    if pd.isna(movie_rating[a]) == False:
      print("a:", a)
      # user similarity score multiply by the movie rating
      score = similar_users[a] * movie_rating[a]
      print("similar_users:", a, similar_users[a])
      # Add the score to the total score for the movie so far
      total += score
      # Add 1 to the count
      count +=1
      # the average score for the item
      movie_score[i] = total / count
  
#%%
# Convert the dictionary to pandas dataframe
movie_score = pd.DataFrame(movie_score.items(), columns=['movie', 'movie_score'])
    
# Sort the movies by movie_score
ranked_movie_score = movie_score.sort_values('movie_score',ascending=False)

#top 10 movies

ranked_movie_score.head(10)
#%%%

# User similarity matrix using cosine similarity
# cosine_similarity cannot take missing values, 
# therefore fill in missing value with 0 before the similarity calculation
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

cosine_user_similar = 1-pairwise_distances(matrix_normaliz.fillna(0), metric="cosine")









