import numpy as np
import csv
import pandas as pd

# Predict via the median number of plays.
from matplotlib import pyplot as plt


train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'global_median.csv'

pro_file = 'profiles.csv'
pro_file = 'profiles_fixed.csv'
artist_file = 'artists.csv'
artist_file = 'artist_feature_matrix.csv'

artist_count_file = 'dict_total_plays_artist.csv'
user_count_file = 'dict_total_plays_listener.csv'

artist_cluster_file = 'clustered_artists.csv'
user_cluster_file = 'clustered_users.csv'


############################### Load the training data.
print("loading training data")
train_data = {}
counter = 0
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        # counter += 1
        # if counter > 1000:
        #     break
        user   = row[0]
        artist = row[1]
        plays  = int(row[2])
        # pairs["%s,%s" % (user,artist)] = 0

        if not user in train_data:
            train_data[user] = {}

        train_data[user][artist] = plays

print("loading profiles")
row_user_dict_index = {}
counter = 0
user_data = {}
with open(pro_file, 'r') as users_fh:
    users_csv = csv.reader(users_fh, delimiter=',', quotechar='"')
    next(users_csv, None)
    for row in users_csv:
        user = row[0]
        data = row[1:]
        user_data[user] = data
        row_user_dict_index[user] = counter
        counter += 1

print("loading artists")
column_artist_dict_index = {}
counter = 0
artist_data = {}
with open(artist_file, 'r') as artists_fh:
    artists_csv = csv.reader(artists_fh, delimiter=',', quotechar='"')
    next(artists_csv, None)
    for row in artists_csv:
        artist = row[0]
        data = row[1:]
        artist_data[artist] = data
        column_artist_dict_index[artist] = counter
        counter += 1


##########################open cluster files
print("loading clusterings 1")
#key_value
art_cluster_dict = {}
cluster_art_dict = {}
with open(artist_cluster_file,'r') as art_cl_fh:
    artCl_csv = csv.reader(art_cl_fh, delimiter=',', quotechar='"')
    rowCounter = -1
    for row in artCl_csv:
        rowCounter +=1
        for artist in row:
            art_cluster_dict[artist] = rowCounter
            if rowCounter in cluster_art_dict:
                cluster_art_dict[rowCounter].append(artist)
            else:
                cluster_art_dict[rowCounter] = [artist]
print("loading clusterings 2")
user_cluster_dict = {}
cluster_user_dict = {}
with open(user_cluster_file,'r') as user_cl_fh:
    userCl_csv = csv.reader(user_cl_fh, delimiter=',', quotechar='"')
    rowCounter = -1
    for row in userCl_csv:
        rowCounter +=1
        for user in row:
            user_cluster_dict[user] = rowCounter
            if rowCounter in cluster_user_dict:
                cluster_user_dict[rowCounter].append(user)
            else:
                cluster_user_dict[rowCounter] = [user]
#key is (user_cluster,artist_cluster)
cluster_interaction_dict = {}
def predict(user,artist):
    user_cluster = user_cluster_dict[user]
    artist_cluster = art_cluster_dict[artist]

    if (user_cluster,artist_cluster) in cluster_interaction_dict:
        #Yippeeeeeeeeee
        out = cluster_interaction_dict[(user_cluster,artist_cluster)]
    else:
        interactions = []
        for close_user in cluster_user_dict[user_cluster]:
            for close_artist in cluster_art_dict[artist_cluster]:
                #print close_user
                #train_data[close_user]
                if close_artist in train_data[close_user]:
                    interactions.append(train_data[close_user][close_artist])
        interSum = 0
        if len(interactions) > 0:
            for i in interactions:
                interSum += i
            interAvg = interSum*1.0/len(interactions)
        else:
            interAvg = -1 ###########################FIXME IM BROKEN use global average#############################################
        cluster_interaction_dict[(user_cluster,artist_cluster)] = interAvg
        out = interAvg
    return out
def predict_a_lot(listUsers,listArtists):
    if len(listUsers) != len(listArtists):
        print "wtf bruh, list sizes are different"
    else:
        predictions = []
        for i in range(len(listUsers)):
            predictions.append(listUsers[i],listArtists[i])
    return predictions
print(predict("306e19cce2522fa2d39ff5dfc870992100ec22d2","4ac4e32b-bd18-402e-adad-ae00e72f8d85"))
exit()






# count = 0
# count_overlap = 0
# with open(test_file, 'r') as train_fh:
#     test_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
#     next(test_csv, None)
#     for row in test_csv:
#         user = row[0]
#         artist = row[1]
#         if "%s,%s" % (user, artist) in pairs:
#             count_overlap +=1
#         count += 1


def exampleEntry(dict,numExamples = 5):
    import random
    examples = []
    keys = dict.keys()
    for i in range(numExamples):
        index = random.randint(0,len(keys))
        print("%s : %s" % (keys[index], dict[keys[index]]))

##################### Load artist and listener specific data
print("loading artist data")
artist_counts = {}
user_counts = {}
#########Artist
with open(artist_count_file, 'r') as artist_fh:
    artist_csv = csv.reader(artist_fh, delimiter=',', quotechar='"')
    #next(artist_csv, None)
    for row in artist_csv:
        artist = row[0]
        plays = int(row[1])
        artist_counts[artist] = plays



##########User
print("loading user data")
with open(user_count_file, 'r') as user_fh:
    user_csv = csv.reader(user_fh, delimiter=',', quotechar='"')
    #next(user_csv, None)
    for row in user_csv:
        user = row[0]
        #print (user)
        #exit()
        plays = int(row[1])
        user_counts[user] = plays





####################### make samples by features matrix
print("make samples by feature")
# each sample is a training interaction
# many features
dictInteractionToIndex = {}
index = -1
totalX = []
totalY = []
# print(user_counts.keys())
listNotIncluded = {}
for user in train_data.keys():
    # if user not in user_counts:
    #     #print ("wtf" + str(index))
    #     if user not in listNotIncluded:
    #         listNotIncluded[user] = 0
    #     else:
    #         listNotIncluded[user] = listNotIncluded[user] + 1
    #     continue
    #print(user_counts[user])
    for artist in train_data[user].keys():
        # if artist not in artist_counts:
        #     #print ("wtf artist" + str(index))
        #     if artist not in listNotIncluded:
        #         listNotIncluded[artist] = 0
        #     else:
        #         listNotIncluded[artist] = listNotIncluded[artist] + 1
        #     continue
        interaction = user, artist
        index += 1
        dictInteractionToIndex[interaction] = index

        rowY = [train_data[user][artist]]
        rowX = []
        # add artist data
        rowX.append(artist_counts[artist])
        for i in artist_data[artist]:
            try:
                rowX.append(float(i))
            except ValueError:
                if i == "Male":
                    rowX.append(1)
                elif i == "Female":
                    rowX.append(0)
            #rowX.append(float(i))
        # add user data
        rowX.append(user_counts[user])
        for i in user_data[user]:
            try:
                rowX.append(float(i))
            except ValueError:
                if i == "m":
                    rowX.append(1)
                elif i == "f":
                    rowX.append(0)


        totalX.append(rowX)
        totalY.append(rowY)



############################# RF
print totalY[1]
print totalX[1]
print("starting RF")
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import cross_val_score

model = RFR(criterion='mae') #USING MAF??

print("training")
#model.fit(totalX,totalY)
scores = cross_val_score(model, totalX, totalY,cv=2,verbose=10)
print(scores)
exit()



############################ Compute the global median.
plays_array = []
for user, user_data in train_data.iteritems():
    for artist, plays in user_data.iteritems():
        plays_array.append(plays)
global_median = np.median(np.array(plays_array))
print "global median:", global_median

# Write out test solutions.
with open(test_file, 'r') as test_fh:
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    next(test_csv, None)

    with open(soln_file, 'w') as soln_fh:
        soln_csv = csv.writer(soln_fh,
                              delimiter=',',
                              quotechar='"',
                              quoting=csv.QUOTE_MINIMAL)
        soln_csv.writerow(['Id', 'plays'])

        for row in test_csv:
            id     = row[0]
            user   = row[1]
            artist = row[2]

            soln_csv.writerow([id, global_median])








######################## sparse matrix stuff


import scipy.sparse as sparse
#sparse.coo_matrix(features, (rows[:], columns[:]))

def make_coo_matrix_from_vector_dict(dict,rowDict):
    datapoints = []
    rows = []
    cols = []
    for i in dict.keys():
        datapoints.append(dict[i])
        rows.append(rowDict[i])
        cols.append(0)
    return sparse.coo_matrix((datapoints,(rows,cols)))

artist_matrix = make_coo_matrix_from_vector_dict(artist_counts, column_artist_dict_index)
user_matrix = make_coo_matrix_from_vector_dict(user_counts, row_user_dict_index)

def make_coo_matrix_from_matrix_dict(dict,rowDict,colDict):
    datapoints = []
    rows = []
    cols = []
    for i in dict.keys():
        for j in dict[i].keys():
            datapoints.append(dict[i][j])
            rows.append(rowDict[i])
            cols.append(colDict[j])
    return sparse.coo_matrix((datapoints,(rows,cols)))
#trainData to matrix


rf_train_Y = make_coo_matrix_from_matrix_dict(train_data, row_user_dict_index, column_artist_dict_index)
print(artist_matrix.get_shape())
print(user_matrix.get_shape())
print(rf_train_Y.get_shape())
rf_train_X =artist_matrix.dot(user_matrix.transpose())
print(rf_train_X.get_shape())



########################