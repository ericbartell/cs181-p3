import numpy as np
import csv

# Predict via the median number of plays.
from matplotlib import pyplot as plt


train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'global_median.csv'

pro_file = 'profiles.csv'
artist_file = 'artists.csv'

artist_count_file = 'dict_total_plays_artist.csv'
user_count_file = 'dict_total_plays_listener.csv'

# Load the training data.
train_data = {}
# pairs = {}
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = int(row[2])
        # pairs["%s,%s" % (user,artist)] = 0
    
        if not user in train_data:
            train_data[user] = {}
        
        train_data[user][artist] = plays

row_user_dict_index = {}
counter = 0
with open(pro_file, 'r') as users_fh:
    users_csv = csv.reader(users_fh, delimiter=',', quotechar='"')
    next(users_csv, None)
    for row in users_csv:
        user = row[0]
        row_user_dict_index[user] = counter
        counter += 1

column_artist_dict_index = {}
counter = 0
with open(artist_file, 'r') as artists_fh:
    artists_csv = csv.reader(artists_fh, delimiter=',', quotechar='"')
    next(artists_csv, None)
    for row in artists_csv:
        artist = row[0]
        column_artist_dict_index[artist] = counter
        counter += 1


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

# Load artist and listener specific data
artist_counts = {}
user_counts = {}
with open(artist_count_file, 'r') as artist_fh:
    artist_csv = csv.reader(artist_fh, delimiter=',', quotechar='"')
    next(artist_csv, None)
    for row in artist_csv:
        artist = row[0]
        plays = int(row[1])
        artist_counts[artist] = plays


with open(user_count_file, 'r') as user_fh:
    user_csv = csv.reader(user_fh, delimiter=',', quotechar='"')
    next(user_csv, None)
    for row in user_csv:
        user = row[0]
        plays = int(row[1])
        user_counts[user] = plays


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


# RF
from sklearn.ensemble import RandomForestRegressor as RFR
model = RFR()

print("training")
model.fit(rf_train_X,rf_train_Y)
print(model.score())
exit()


# Compute the global median.
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
