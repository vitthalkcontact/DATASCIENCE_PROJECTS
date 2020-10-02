# DBSCAN Clustering
##https://github.com/krishnaik06/DBSCAN-Algorithm
# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=3,min_samples=4)

# Fitting the model

model=dbscan.fit(X)
print(model)

labels=model.labels_
print(labels)

from sklearn import metrics

#identifying the points which makes up our core points
sample_cores=np.zeros_like(labels,dtype=bool)

sample_cores[dbscan.core_sample_indices_]=True

#Calculating the number of clusters

n_clusters=len(set(labels))- (1 if -1 in labels else 0)
print(n_clusters)


print(metrics.silhouette_score(X,labels))

##The silhouette ranges from −1 to +1, 
#where a high value indicates that the object is well matched to its own cluster 
#and poorly matched to neighboring clusters.

