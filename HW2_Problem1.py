import numpy as np 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans

X = np.load('./data/p1/X.npy')
X_tr = np.log2(X + 1)
X_tr = X
#pca = PCA(n_components=100)
#pca.fit(X)
#components = (np.cumsum(pca.explained_variance_ratio_) < 0.85).astype('int')

#print(sum(components)+1)
pca = PCA()
Z = pca.fit_transform(X_tr)
components = (np.cumsum(pca.explained_variance_ratio_) < 0.85).astype('int')

print(sum(components)+1)

plt.scatter(X_tr[:, 0], X_tr[:, 1])
plt.title("Scatter plot of 1st and 2nd dimensions", size=10)
plt.xlabel("x1", size=14)
plt.ylabel("x2", size=14)
plt.axis("equal")
plt.show()

plt.scatter(Z[:, 0], Z[:, 1])
plt.title("Scatter plot of 1st and 2nd Principal Components", size=10)
plt.xlabel("z1", size=14)
plt.ylabel("z2", size=14)
plt.axis("equal")
plt.show()

mds = MDS(n_components=2, verbose=1, eps=1e-5)
mds.fit(X_tr)
plt.scatter(mds.embedding_[:,0], mds.embedding_[:,1])
plt.title("MDS Plot", size=18)
plt.axis("equal")
plt.show()

tsne = TSNE(n_components=2, perplexity=40)
z_tsne = tsne.fit_transform(Z[:, 0:50])
plt.scatter(z_tsne[:, 0], z_tsne[:, 1])
plt.title("TSNE, perplexoty=40", size=18)
plt.axis("equal")
plt.show()

kmeans = KMeans(n_clusters=4, n_init=100)
y = kmeans.fit_predict(Z)
plt.scatter(Z[:, 0], Z[:, 1], c=y)
plt.title("Scatter plot of 1st and 2nd Principal Components", size=10)
plt.xlabel("z1", size=14)
plt.ylabel("z2", size=14)
plt.axis("equal")
plt.show()

plt.scatter(mds.embedding_[:,0], mds.embedding_[:,1], c=y)
plt.title("MDS Plot", size=18)
plt.axis("equal")
plt.show()

plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y)
plt.title("TSNE, perplexoty=40", size=18)
plt.axis("equal")
plt.show()

all_kmeans = [KMeans(n_clusters=i+1, n_init=100) for i in range(8)]
for i in range(8):
    all_kmeans[i].fit(Z[:, 0:50])
inertias = [all_kmeans[i].inertia_ for i in range(8)]
print(inertias[3])

plt.plot(np.arange(1,9), inertias)
plt.title("KMeans Sum of Squares Criterion")
plt.xlabel("# Clusters", size=14)
plt.ylabel("Within-Cluster Sum of Squares")
plt.show()

kmeans = KMeans(n_clusters=5, n_init=100)
y = kmeans.fit_predict(X_tr)
means = np.array([np.mean(X_tr[np.where(y==i)[0]],0) for i in range(5)])
mds_means = MDS(n_components=2)
mds_means.fit(means)

plt.scatter(Z[:, 0], Z[:, 1])
plt.title("Scatter plot of 1st and 2nd Principal Components", size=10)
plt.xlabel("z1", size=14)
plt.ylabel("z2", size=14)
plt.axis("equal")
plt.show()

plt.scatter(mds_means.embedding_[:,0], mds_means.embedding_[:,1], 
            c=[i for i in range(5)], s=200)
plt.title("MDS Plot", size=18)
plt.axis("equal")
plt.show()

tsne = TSNE(n_components=2, perplexity=40)
z_tsne = tsne.fit_transform(Z[:, 0:50])
plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y)
plt.title("TSNE, perplexity=40", size=18)
plt.axis("equal")
plt.show()


