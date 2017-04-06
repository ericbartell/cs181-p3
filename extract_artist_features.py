import pandas as pd
import musicbrainzngs as mb
import time
import collections

# I hate unicode. This should recursively convert the nested mess of unicode
# that is the output of this into str so that I can write this to a csv
def convert(data):
    if isinstance(data, basestring):
        return data.encode('utf-8')
    elif isinstance(data, collections.Mapping):
        return dict(map(convert, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert, data))
    else:
        return data

artists = pd.read_csv("artists.csv")
# Note, there are at least 10 artists with no name, and invalid ids
# there are potentially even more artists with bad ids

feature_lst = []
# this will be a list of dictionaries, one per artist, {} for invalid ids

mb.set_useragent("HW assignment", "2.0.6", "hgold@g.harvard.edu")
# have to tell the database how to contact me if I screw up majorly
# they won't let you extract any info if you skip this


rate_limit = 50 # currently I think I can submit 50 requests per second
seconds = 1.5   # wait a bit longer than a second, just to be safe

for index in range(artists.shape[0]):
    try:
        # try to extract the features for each artist
        f_dict = mb.get_artist_by_id(artists.artist[index])['artist']
        feature_lst.append(f_dict)
    except mb.ResponseError as e:
        feature_lst.append({})
        print e
        # 404 not found errors are fine, but I need to catch them anyways
        # what I don't want to see are any rate-throttling errors

    if (index % rate_limit) == 0:
        print index
        time.sleep(seconds)

# Of course, each of these dictionaries has their own, unique set of keys
# so I'm converting it to a dataframe so each key -> column and it fills
# empty cells with NaN/None automatically
### the convert function transforms the unicode -> str
feature_df = pd.DataFrame(convert(feature_lst))

print feature_df.info() # mostly cause I want an overview of how sparse this is

# Output dataframe to csv, will load and parse this monster in another script
feature_df.to_csv("mb_artist_features.csv")

