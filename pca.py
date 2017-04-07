import numpy as np
from sklearn.decomposition import IncrementalPCA
#import pandas as pd
import sys

user_plays_file = open(sys.argv[1],'r')
data = []

print "creating matrix"
count = 0
for line in user_plays_file:

	line = line.rstrip().split(',')[1:]
	line = [int(x) for x in line]
	data.append(line)

	count += 1
	#if count > 20000:
	#	break

X = np.array(data)
#print X

print "running PCA"
pca = IncrementalPCA(n_components=10)
pca.fit(X)
print(pca.explained_variance_ratio_) 
X_transform = pca.transform(X)
for line in X_transform:
	
	line = [str(x) for x in line]
	print ','.join(line)
