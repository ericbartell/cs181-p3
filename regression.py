
# coding: utf-8

# In[2]:

#This file was used to generate the author features
#Imports
import numpy as np
import csv


# In[3]:

# File Names
train_file = 'train.csv'
test_file  = 'test.csv'
profile_file = 'profiles_fixed.csv'
artist_file = 'artist_feature_matrix.csv'
#soln_file  = 'user_median.csv'


# In[4]:

# Load the training data.
# training data is saved as a dictionary inside a dictionary 
# [user][artists] = plays
train_data = {}
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = row[2]
    
        if not user in train_data:
            train_data[user] = {}
        
        train_data[user][artist] = int(plays)


# In[5]:

#Load profile data and author data using the newly defined features
profile_data = {}
with open(profile_file, 'r') as profile_fh:
    profile_csv = csv.reader(profile_fh, delimiter=',', quotechar='"')
    next(profile_csv, None)
    for row in profile_csv:
        user = row[0]
        
        sex = row[1]
        if sex == 'm':
            sex = 1
        elif sex == 'f':
            sex = 2
        else:
            sex = 3
        
        age = row[2]
        try:
            age = int(age)
        except:
            age = -1
        
        countries = {}
        count = 0
        country = row[3]
        if country in countries.keys():
            country = countries[country]
        else:
            count += 1
            countries[country] = count
            country = countries[country]
        
        profile_data[user] = [sex, age, country]
        
author_data = {} 
with open(artist_file, 'r') as artist_fh:
    artist_csv = csv.reader(artist_fh, delimiter=',', quotechar='"')
    for row in artist_csv:
        user = row[0]
        
        final_row = []
        #Convert sex to int
        sex = row[1]
        if sex == 'm':
            sex = 1
        elif sex == 'f':
            sex = 2
        else:
            sex = 3
        final_row.append(sex)

        #Convert group to int
        groups = {}
        count = 0
        value = row[2]
        if value in groups.keys():
            value = groups[value]
        else:
            count += 1
            groups[value] = count
            value = groups[value]
        final_row.append(value)
        
        #Convert ended to int
        groups = {}
        count = 0
        value = row[3]
        if value in groups.keys():
            value = groups[value]
        else:
            count += 1
            groups[value] = count
            value = groups[value]
        final_row.append(value)

        final_row.append(int(float(row[4])))
        final_row.append(int(float(row[5])))
        final_row.append(int(float(row[6])))

        #Convert country to int
        groups = {}
        count = 0
        value = row[7]
        if value in groups.keys():
            value = groups[value]
        else:
            count += 1
            groups[value] = count
            value = groups[value]
        final_row.append(value)

        final_row.extend([int(item) for item in row[8:435]])
        author_data[user] = final_row


# In[6]:

X = []
Y = []
count = 0
def create_training_matrix():
    for key,val in train_data.iteritems():
        for artist,label in val.iteritems():
            X.append(np.array(profile_data[key] + author_data[artist]))
            Y.append(int(label))

create_training_matrix()


# In[7]:

feat = np.array(X[0:4000000])
labels = np.array(Y[0:4000000])
print len(X)

print feat.shape
print labels.shape


# In[14]:

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn import metrics

predictor= RandomForestRegressor()
scores = cross_val_score(predictor, feat, labels, scoring='neg_mean_absolute_error', cv=3, verbose=2)
print scores


# In[ ]:

from sklearn.model_selection import cross_val_score
from sklearn import datasets, linear_model
predictor= linear_model.LinearRegression()
predictor.fit(feat, labels)

#scores = cross_val_score(predictor, feat, labels, scoring='neg_mean_absolute_error', cv=3, verbose=2)
#print scores


# In[ ]:



