# Agglomerative hierarchical clustering 

import pandas
import matplotlib.pyplot as pyplot

import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering


dataset = pandas.read_csv("dataset_ahc.csv")

print(dataset)

pyplot.scatter(dataset['x1'], dataset['x2'])
pyplot.savefig("scatterplot_ahc.png")
pyplot.close()


pyplot.title("Dendrogram")
dendrogram_object = shc.dendrogram(shc.linkage(dataset, method="ward"))
pyplot.savefig("dendrogram_ahc.png")
pyplot.close()



machine  = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
results_ahc = machine.fit_predict(dataset)

pyplot.scatter(dataset['x1'], dataset['x2'], c=results_ahc)
pyplot.savefig("scatterplot_ahc_color.png")
pyplot.close()




dataset = pandas.read_csv("dataset_moon.csv")

print(dataset)

pyplot.scatter(dataset['x1'], dataset['x2'])
pyplot.savefig("scatterplot_moon.png")
pyplot.close()


pyplot.title("Dendrogram")
dendrogram_object = shc.dendrogram(shc.linkage(dataset, method="single"))
pyplot.savefig("dendrogram_moon.png")
pyplot.close()



machine  = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="single")
results_moon = machine.fit_predict(dataset)

pyplot.scatter(dataset['x1'], dataset['x2'], c=results_moon)
pyplot.savefig("scatterplot_moon_color.png")
pyplot.close()




dataset = pandas.read_csv("dataset_new_moon.csv")

print(dataset)

pyplot.scatter(dataset['x1'], dataset['x2'])
pyplot.savefig("scatterplot_new_moon.png")
pyplot.close()


pyplot.title("Dendrogram")
dendrogram_object = shc.dendrogram(shc.linkage(dataset, method="single"))
pyplot.savefig("dendrogram_new_moon.png")
pyplot.close()



machine  = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="single")
results_new_moon = machine.fit_predict(dataset)

pyplot.scatter(dataset['x1'], dataset['x2'], c=results_new_moon)
pyplot.savefig("scatterplot_new_moon_color.png")
pyplot.close()



