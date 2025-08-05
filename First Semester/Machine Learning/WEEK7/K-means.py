import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from sklearn_extra.cluster import KMedoids

np.random.seed(0)
S1 = np.random.normal(loc=[10, 3333], scale=[5, 516], size=(35, 2)) # loc represents the mean, scale is standard deviation, size is shape of the output array
S2 = np.random.normal(loc=[45, 4933], scale=[5, 1516], size=(35, 2))
S3 = np.random.normal(loc=[70, 8333], scale=[5, 616], size=(35, 2))
X = np.concatenate((S1, S2, S3))

np.random.seed(4)
X += np.random.randn(len(X), 2) * [20.1] # we add a scaling noise to both data with same scale

# K-Means if we use library (practice implement without library)
k_means = KMeans(n_clusters=3, random_state=5)
#k_medoids = KMedoids(n_clusters=3, random_state=5) if you wan to run KMedoids (you need to fix rest of code)
k_means.fit(X)


# Plot the data and centroids
colors = ['blue', 'red', 'green', 'orange', 'black']

plt.scatter(X[:, 0], X[:, 1], c=[colors[label] for label in k_means.labels_])
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], marker='x', s=300, c='black', linewidths=1.5)
plt.title('K-Means Clustering (k=4)')
plt.xlabel('Age')
plt.ylabel('Monthly Income ($)')

plt.show()
