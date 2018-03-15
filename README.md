# kmeans-kohonen

K-means clustering algorithm:
 
* Choose k initial centroids in the space of the input dataset 
* Repeat:
    * E-step: put each pattern from the input dataset in a cluster defined by its nearest centroid. 
    * M-step: move each centroid so to minimize its distance from all the patterns in that cluster.
