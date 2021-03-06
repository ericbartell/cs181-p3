import pandas as pd
import csv

pairs = pd.read_csv("train.csv")

# initialize a dictionary keyed on users, with 'total plays' set to 0
user_total_plays = dict.fromkeys(pairs.user.unique(), 0)

artist_total_plays = dict.fromkeys(pairs.artist.unique(), 0)

for row in pairs.itertuples():
    # row = (index, user, artist, plays)
    user = row[1]
    artist = row[2]
    plays = row[3]
    user_total_plays[user] += plays
    artist_total_plays[artist] += plays

# Now write the dictionary to a file, so we don't have to recalculate it:
#f = open("dict_total_plays.txt", 'w')
#f.write(str(user_total_plays))
#f.close()
### actually, this text file is ugly. let's output a csv
w = csv.writer(open("dict_total_plays_listener.csv", 'w'))
for key, val in user_total_plays.items():
    w.writerow([key, val])

w = csv.writer(open("dict_total_plays_artist.csv", 'w'))
for key, val in artist_total_plays.items():
    w.writerow([key, val])


