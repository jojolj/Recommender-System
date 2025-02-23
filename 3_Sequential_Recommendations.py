#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 00:01:05 2023

@author: kailinzhang
"""

import numpy as np
import pandas as pd
ratings = pd.read_csv('/Users/kailinzhang/Downloads/ml-latest-small/ratings.csv')
movies = pd.read_csv('/Users/kailinzhang/Downloads/ml-latest-small/movies.csv')
df = pd.merge(ratings, movies, on='movieId', how='inner')
matrix = df.pivot_table(index='userId', columns='title', values='rating')
matrix_normaliz = matrix.subtract(matrix.mean(axis=1), axis = 'rows')
user_similarity = matrix_normaliz.T.corr()
users_num = ratings['userId'].max()
rating_mean = []
for i in range(users_num):
    mean = ratings[ratings['userId']==i+1]['rating'].mean()
    rating_mean.append(mean)
def predictions(userId,movieId):
    # For given movieId, get all users who rate it.
    r_b = ratings[ratings['movieId']==movieId]['userId']
    # For given movieId, get all ratings of known users
    r_b_p = ratings[ratings['movieId']==movieId]['rating']
    # If the movie has no rate, the prediction cannot be calculated
    if r_b.empty:
        return 0
    else:
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
        if sum_sim == 0:
            return 0
        else:
            # Calculate the prediction
            pre = rating_mean[userId - 1] + (sum / sum_sim)
        return(pre)

def allratings(userId):
    pred_i = []
    for i in range(len(movies['movieId'])):
        tem_movieId = movies['movieId'][i]
        # If the rating has been recorded, use this rating
        if not ratings[(ratings['userId']==userId) & (ratings['movieId']==tem_movieId)]['rating'].empty:
            pred = ratings[(ratings['userId']==userId) & (ratings['movieId']==tem_movieId)]['rating']
        # If the rating has not been recorded, predict it
        else:
            pred = predictions(userId,tem_movieId)
        #relavant_movie = np.append(relavant_movie, tem_movieId)
        pred_i = np.append(pred_i, pred)
        
    return pred_i
#%%
def sequentialrec(users,k,n_iteration):
    group_sat_overall = []
    vstack = np.zeros((len(movies),))
    for i in users:
        ratings_i = allratings(i)
        vstack = np.vstack((vstack, ratings_i))
    vstack = np.delete(vstack,0,axis=0).T
    mean = np.mean(vstack, axis=1)
    min = np.min(vstack, axis=1)
    top_k = np.zeros((k,))
    for i in users:
        ratings_i = allratings(i)
        top_k_i = np.argsort(ratings_i)[::-1][:k]
        top_k = np.vstack((top_k,top_k_i))
    top_k = np.delete(top_k,0,axis=0).T
    alpha_j = [0.5]
    satoverall = np.zeros((1,len(users)))
    for j in range(1,n_iteration+1):
        if j > n_iteration:
            break
        score = []
        
        for i in range(len(vstack)):
            score_i = ((1 - alpha_j[j-1]) * mean[i]) + (alpha_j[j-1] * min[i])
            score = np.append(score, score_i)
        grouprec = np.argsort(score)[::-1][:k]
        
        grouplistsat_i = 0
        userlistsat_i = 0
        grouplistsat = []
        userlistsat = []
        for i in range(len(users)):
            for l in grouprec:
                sat_i = allratings(users[i])[l]
                grouplistsat_i += sat_i
            grouplistsat = np.append(grouplistsat,grouplistsat_i)
        
        for i in range(len(users)):
            for l in top_k[:,i]:
                usersat_i = allratings(users[i])[int(l)]
                userlistsat_i += usersat_i
            userlistsat = np.append(userlistsat,userlistsat_i)
        
        sat = grouplistsat/userlistsat
        
        top_10_recommendations = []
        for i in grouprec:
            recommendation = movies.loc[i]['title']
            top_10_recommendations = np.append(top_10_recommendations, recommendation)
        print(top_10_recommendations)
        
        group_sat = sat.sum()/len(users)
        group_sat_overall = np.append(group_sat_overall, group_sat)
        
        alpha_j_ = sat.max() - sat.min()
        alpha_j = np.append(alpha_j, alpha_j_)
        
        satoverall += sat
    return(group_sat_overall)
users = [6,7,8]
k = 10
n_iteration = 3
sequentialrec(users,k,n_iteration)
#%%
predictions(1,10)