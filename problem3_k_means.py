# CS 181, Spring 2017
# Homework 4: Clustering
# Name:
# Email:

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv

from sklearn.cluster import KMeans

train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'global_median.csv'

pro_file = 'profiles.csv'
pro_file = 'profiles_fixed.csv'
artist_file = 'artists.csv'
artist_file = 'artist_feature_matrix.csv'

artist_count_file = 'dict_total_plays_artist.csv'
user_count_file = 'dict_total_plays_listener.csv'

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

column_user_dict_index = {}
counter = 0
user_data = {}
with open(pro_file, 'r') as user_fh:
	user_csv = csv.reader(user_fh, delimiter=',', quotechar='"')
	next(user_csv, None)
	for row in user_csv:
		user = row[0]
		data = row[1:]
		user_data[user] = data
		column_user_dict_index[user] = counter
		counter += 1


#for i in artist_data.keys()[0:20]:
	#print artist_data[i][0:50]

def getUniques(data, columnIndex):
	uniqueVals = {}
	counter = 0
	for key in data.keys():
		interestingCol = data[key][columnIndex]
		if interestingCol not in uniqueVals:
			uniqueVals[interestingCol] = counter
			counter += 1

	return uniqueVals

#print(getUniques(artist_data,6))
#for i in [0,1,2,6]:
	#pass
def make_one_hot(data, cols):
	columnToColumns = {}
	newData = {}
	columnSets = {}

	uniqueDict = {}


	first = True
	for key in data.keys():
		oldRow = data[key]
		if first:
			first = False
			for i in range(len(oldRow)):
				columnToColumns[i] = []
		newRow =[]
		currentIndex = -1

		for i in range(len(oldRow)):
			if i in cols:
				if i in uniqueDict:
					uniques = uniqueDict[i]
				else:
					uniques = getUniques(data,i)
					uniqueDict[i] = uniques
				for j in range(len(uniques.keys())):
					currentIndex += 1
					if oldRow[i] == uniques.keys()[j]:
						newRow.append(1)
					else:
						newRow.append(0)
					columnToColumns[i].append(currentIndex)
			else:
				currentIndex += 1
				newRow.append(float(oldRow[i]))
				columnToColumns[i].append(currentIndex)
		newData[key] = newRow
	#print(columnToColumns)

	first = True
	for key in newData.keys():
		if first:
			maxRow = []
			minRow = []
			first = False
			for i in range(len(newData[key])):
				maxRow.append(0)

				minRow.append(float("inf"))

		for i in range(len(newData[key])):
			if newData[key][i] > maxRow[i]:
				maxRow[i] = newData[key][i]
			if newData[key][i] < minRow[i]:
				minRow[i] = newData[key][i]
	#print maxRow
	for key in newData.keys():
		newRow = []
		for i in range(len(newData[key])):
			newRow.append((newData[key][i]*1.0-minRow[i])/(maxRow[i]-minRow[i]))
		newData[key] = newRow
		#print newRow


	return newData

one_hot_data_artist = make_one_hot(artist_data,[0,1,2,6])
def regularize_OneHot(onehot):
	listKeys = onehot.keys()
	index = -1
	keyToRow = {}
	matrix = []
	for i in listKeys:
		index +=1
		keyToRow[i] = index
		matrix.append(onehot[i])
	npmatrix = np.matrix(matrix)
	return npmatrix,listKeys
artist_matrix,artist_key_list = regularize_OneHot(one_hot_data_artist)




print("regularized")
K = 30
kmeans = KMeans(n_clusters=K,random_state=0).fit(artist_matrix)
clusters = [[]for i in range(K)]

for i in range(len(kmeans.labels_)):
	clusters[kmeans.labels_[i]].append(artist_key_list[i])
w = csv.writer(open("clustered_artists.csv", 'w'))
for row in clusters:
    w.writerow(row)


#print(kmeans.cluster_centers_)



one_hot_data_user = make_one_hot(user_data,[0,2])
user_matrix,user_key_list = regularize_OneHot(one_hot_data_user)
#def dist_between_artists(id1, id2):
print("regularized")
K = 100
kmeans = KMeans(n_clusters=K,random_state=0,verbose=10).fit(user_matrix)
clusters = [[] for i in range(K)]

for i in range(len(kmeans.labels_)):
	clusters[kmeans.labels_[i]].append(user_key_list[i])
w = csv.writer(open("clustered_users.csv", 'w'))
for row in clusters:
    w.writerow(row)






class KMeans(object):
	# K is the K in KMeans
        def __init__(self, K):
		self.K = K
		self.pickedCentroid = []

	def makeCentroid(self):
		index = -1
		while index == -1 or index in self.pickedCentroid:
			index = np.random.randint(0,len(self.data))
		
		self.pickedCentroid.append(index)
		return self.data[index]
	# X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.

	#def pointDist(self,point1,point2):


	def fit(self, X, numIter = 20):
		self.data = X
		self.centroids = []
		picked = []
		for i in range(self.K):
			newCentroid = self.makeCentroid()
			self.centroids.append(newCentroid)
		for i in range(numIter):
			#assign data to centroid
			self.picIDs = []
			for pic in self.data:
				#print pic
				minCentIndex = -1
				minDist = -1
				for centroidIndex in range(self.K):

					distToCentroid = np.sum(np.square(self.centroids[centroidIndex] - pic))
					if minDist == -1 or distToCentroid<minDist:
						minDist = distToCentroid
						minCentIndex = centroidIndex
				self.picIDs.append(minCentIndex)
			groups = [[] for i in range(self.K)]
			for i in range(len(self.picIDs)):
				groups[self.picIDs[i]].append(i)

			#move centroids
			newCentroids = []
			for i in range(self.K):
				rep = np.zeros(len(self.data[0]))
				for memberIndex in groups[i]:
					rep += data[memberIndex]
				rep /= (1.0*len(groups[i]))
				newCentroids.append(rep)
			self.centroids = newCentroids
			#print(self.centroids)
		pass

	# This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
	def get_mean_images(self):
		return self.centroids

	# This should return the arrays for D images from each cluster that are representative of the clusters.
	def get_representative_images(self, D):
		# assign data to centroid
		self.picIDs = []
		for pic in self.data:
			minCentIndex = -1
			minDist = -1
			for centroidIndex in range(self.K):
				distToCentroid = np.sum(np.square(self.centroids[centroidIndex] - pic))
				if minDist == -1 or distToCentroid < minDist:
					minDist = distToCentroid
					minCentIndex = centroidIndex
			self.picIDs.append(minCentIndex)
		groups = [[] for i in range(self.K)]
		for i in range(len(self.picIDs)):
			groups[self.picIDs[i]].append(i)

		output = []
		for i in range(self.K):
			for j in range(D):
				output.append(data[groups[i][np.random.randint(0,len(groups[i]))]])

		return output

	def get_objective(self):
		totalDist = 0
		for picID in range(len(self.data)):
			distToCentroid = np.sum(np.square(self.centroids[self.picIDs[picID]] - self.data[picID]))
			totalDist += distToCentroid
		return totalDist/(1.0*len(self.data))
	# img_array should be a 2D (square) numpy array.
	# Note, you are welcome to change this function (including its arguments and return values) to suit your needs. 
	# However, we do ask that any images in your writeup be grayscale images, just as in this example.
	def create_image_from_array(self, img_array):
		plt.figure()
		plt.imshow(img_array, cmap='Greys_r')
		plt.show()
		return

	def create_image_from_array_array(self, img_array_array,rep = False):
		if rep:
			f,axarr = plt.subplots(len(img_array_array)/self.K,self.K)

			for i in range(len(img_array_array)):
					axarr[i%2,i/2].imshow(img_array_array[i], cmap='Greys_r')
					axarr[i%2,i/2].set_yticklabels([])
					axarr[i%2,i/2].set_xticklabels([])
		else:
			f,axarr = plt.subplots(len(img_array_array))
			for i in range(len(img_array_array)):
				axarr[i].imshow(img_array_array[i], cmap='Greys_r')
				axarr[i].set_yticklabels([])
				axarr[i].set_xticklabels([])

		#plt.figure()

		return plt


# This line loads the images for you. Don't change it! 
#pics = np.load("images.npy", allow_pickle=False)

# You are welcome to change anything below this line. This is just an example of how your code may look.
# That being said, keep in mind that you should not change the constructor for the KMeans class, 
# though you may add more public methods for things like the visualization if you want.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.


K = 10
KMeansClassifier = KMeans(K=10)#, useKMeansPP=False)
KMeansClassifier.fit(artist_matrix)
exit()
objectives = []
numIters = [5,10,20,30,50,100]
for i in numIters:
	print(i)
	KMeansClassifier = KMeans(K=10)  # , useKMeansPP=False)
	KMeansClassifier.fit(data,numIter=i)
	objectives.append(KMeansClassifier.get_objective())
plt.scatter(numIters,objectives)
plt.savefig("objectiveScatter.png")

# for i in [5,10,20]:
# 	for restart in [1,2,3]:
# 		KMeansClassifier = KMeans(K=i)  # , useKMeansPP=False)
# 		KMeansClassifier.fit(pics)
# 		print("K%s_mean%s scores: %s" % (i,restart,KMeansClassifier.get_objective()))
# 		KMeansClassifier.create_image_from_array_array(KMeansClassifier.get_mean_images()).savefig("K%s_mean%s.png" % (i,restart))
# 	KMeansClassifier.create_image_from_array_array(KMeansClassifier.get_representative_images(2),rep=True).savefig("K%s_rep.png" % (i))




