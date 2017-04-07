import numpy as np
import csv
import json
import urllib2
import pprint
import math
import pickle

# Load the training data.
# training data is saved as a dictionary inside a dictionary
# [user][artists] = plays
def load_training(train_file):
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
    return train_data

# Load the User Profiles as a Dictionary
def load_profile(profile_file):
    profile_data = {}
    with open(profile_file, 'r') as profile_fh:
        profile_csv = csv.reader(profile_fh, delimiter=',', quotechar='"')
        next(profile_csv, None)
        for row in profile_csv:
            user = row[0]
            sex = row[1]
            age = row[2]
            country = row[3]

            profile_data[user] = [sex, age, country]
    return profile_data

# Load Artist Profiles as a Dictionary
def load_artist(artist_file):
    artist_data = {}
    with open(artist_file, 'r') as artist_fh:
        artist_csv = csv.reader(artist_fh, delimiter=',', quotechar='"')
        next(artist_csv, None)
        for row in artist_csv:
            user = row[0]
            name = row[1]

            artist_data[user] = name
    return artist_data

class artist_features():
    def __init__(self,artist_file):
        self.artist_data = load_artist(artist_file)
        self.artists = self.artist_data.keys()
        self.corrections = {}
        self.artist_features = {}
        self.artist_tags = {}
        self.filtered_tags = {}

    #We create a dictionary of the broken link and the new link
    def create_corrections_dict(self):
        self.corrections['0f3515b0-75c9-46c9-b26c-4cd05d26eae7'] = '71f754c0-f2d1-4a54-8d70-cc0ee409ca00'
        self.corrections['10b7b68c-390d-469a-915b-40bac704f288'] = '79b1f58c-ac99-40de-8e5c-a86f6b340bf2'
        self.corrections['4b179fe2-dfa5-40b1-b6db-b56dbc3b5f09'] = '7f4b5546-a3c3-4e55-ba8d-53bf937fed30'
        self.corrections['5385c403-1c49-4f2f-9b98-7085b5c84371'] = '27a8ce8c-4e41-4a56-ba65-3a82662c9af9'
        self.corrections['5aca3051-afa2-4f5c-9974-cc9418482a58'] = 'be345ce7-4bc4-4f8d-8705-72ee5d0fac01'
        self.corrections['64b86e99-b6ec-4fb1-a5cd-f95482d3b57a'] = '2ff63f00-0954-4b14-9007-e19b822fc8b2'
        self.corrections['8f3f7fec-cabf-4366-9c31-06f204b402f5'] = 'b91c664b-5eee-4536-8b70-b1f2be519ac0'
        self.corrections['9bf79f68-c064-44a1-8c2c-5764f1d7c016'] = 'fc086e05-6cc0-4b23-9476-792f423dd0bf'
        self.corrections['ae681605-2801-4120-9a48-e18752042306'] = 'f59c5520-5f46-4d2c-b2c4-822eabf53419'
        self.corrections['b5da400c-9a62-4686-b6fe-91518e57ce5d'] = 'a9126556-f555-4920-9617-6e013f8228a7'
        self.corrections['f1a95c6b-fb2a-41a6-bfcb-2453fee2a38c'] = 'f6580b26-77f7-4a7b-8513-6db0476e8f21'

    def get_features(self):
        #Extracting Artist Features using MusicBrainz API
        count = 1
        for artist in self.artists:
            flag = False
            iteration = 0
            while (flag == False):
                try:
                    url = "http://musicbrainz.org/ws/2/artist/"+artist+"?inc=tags+ratings+works&fmt=json"
                    if artist in self.corrections.keys():
                        url = "http://musicbrainz.org/ws/2/artist/"+corrections[artist]+"?inc=tags+ratings+works&fmt=json"
                    data = json.load(urllib2.urlopen(url))
                    flag = True
                except:
                    flag = False
                    if iteration == 200:
                        print "INVALID" , artist, self.artist_data[artist]
                        url = "http://musicbrainz.org/ws/2/artist/f82bcf78-5b69-4622-a5ef-73800768d9ac?inc=tags+ratings+works&fmt=json"
                        data = json.load(urllib2.urlopen(url))
                        flag = True

            a_gender = data['gender']
            a_type = data['type']
            a_lifespan_ended = data['life-span']['ended']

            try:
                a_lifespan_begin = round(int(data['life-span']['begin'].split('-')[0]),-1)
            except:
                a_lifespan_begin = 2000
                print "LIFESPAN IS NULL"

            try:
                a_lifespan_end = round(int(data['life-span']['end'].split('-')[0]), -1)
            except:
                if a_lifespan_ended == False:
                    a_lifespan_end = 2020
                else:
                    a_lifespan_ended = 2010

            a_lifespan_range = a_lifespan_end - a_lifespan_begin

            try:
                a_country = data['country']
            except:
                a_country = 'US'

            a_works = len(data['works'])

            word_dict = {}
            for item in data['tags']:
                word_dict[item['name'].lower()] = int(item['count'])

            feature_list = [a_gender, a_type, a_lifespan_ended, a_lifespan_begin,                         a_lifespan_end, a_lifespan_range, a_country, a_works]
            self.artist_features[artist] = feature_list
            self.artist_tags[artist] = word_dict

            print count, artist, self.artist_data[artist], data['name']+": ",feature_list, word_dict.keys()
            print ""
            count +=1

    def create_feature_dict(self):
        #Dictionary stores the counts of all words in the document
        words = {}
        for document in artist_tags.values():
            for word in document.keys():
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1

        #Removing all words that are included only once
        filtered_words = { k:v for k, v in words.items() if v>3}
        print len(filtered_words)

        all_words = filtered_words.keys()
        for artist,document in artist_tags.iteritems():
            temp_word_vec = []
            for word in all_words:
                if word in document:
                    temp_word_vec.append(1)
                else:
                    temp_word_vec.append(0)
            self.artist_keys[artist] = np.array(temp_word_vec)

    def create_csv(self):
        with open('artist_feature_matrix.csv', 'wb') as csvfile:
            my_writer = csv.writer(csvfile, delimiter=',')
            for artists,words in self.artist_keys.iteritems():
                row = []
                row.append(artists)
                for item in self.artist_features[artists]:
                    if item == None:
                        row.append('na')
                    else:
                        row.append(item)
                row.extend(words)
                my_writer.writerow(row)

    def save_pickle(self,feature_name, tag_name):
        pickle.dump(self.artist_features, open(feature_name, "wb" ) )
        pickle.dump(self.artist_tags , open( tag_name, "wb" ) )

    def load_pickle(self,feature_name, tag_name):
       self.artist_features = pickle.load(open(feature_name, "rb"))
       self.artist_tag = pickle.load(open(artist_tag, "rb"))

    def find_invalid(self):
        count = 1
        for artist in self.artists:
            flag = False
            iteration = 0
            while (flag == False):
                try:
                    url = "http://musicbrainz.org/ws/2/artist/"+artist+"?inc=tags+ratings+works&fmt=json"
                    data = json.load(urllib2.urlopen(url))
                    flag = True
                except:
                    flag = False
                    iteration +=1
                    if iteration == 100:
                        print ""
                        print "INVALID" , artist, self.artist_data[artist]
                        flag = True
            print count,
            count +=1

if __name__=="__main__":
    artist_file = 'artists.csv'
    feat_gen = artist_features(artist_file)
    #feat_gen.create_corrections_dict()
    #feat_gen.get_features()
    feat_gen = create_feature_dict()
    feat_gen = create_csv()

