
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import sys

user_file = open(sys.argv[1],'r')
mat = []

for line in user_file:

	plays = line.rstrip().split(',')
	#plays = [int(x) for x in plays]
	mat.append(plays)

X = np.array(mat)
#print X
kmeans = KMeans(n_clusters = 20, random_state=0).fit(X)
labels = kmeans.labels_
for label in labels:
	print (label)
