# #k-means clustering algorithm
# from sknetwork.clustering import KMeans
# from sknetwork.data import karate_club
# kmeans = KMeans(n_clusters=3)
# adjacency = karate_club()
# labels = kmeans.fit_transform(adjacency)
# print(len(set(labels)))

# louvain/dugue
from sknetwork.clustering import Louvain
from sknetwork.data import karate_club
louvain = Louvain()
# Undirected graph with 34 nodes, 78 edges and 2 labels
# https://en.wikipedia.org/wiki/Zachary%27s_karate_club
adjacency = karate_club()
labels = louvain.fit_transform(adjacency)
print(len(set(labels)))