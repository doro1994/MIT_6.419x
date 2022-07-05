import numpy as np 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

X = np.load('./data/p2_unsupervised_reduced/X.npy')
X_tr = np.log2(X + 1)
X_eval_test =np.load('./data/p2_evaluation_reduced/X_test.npy') 
X_eval_test_tr = np.log2(X_eval_test + 1)
X_eval_train =np.load('./data/p2_evaluation_reduced/X_train.npy') 
X_eval_train_tr = np.log2(X_eval_train + 1)
y_eval_test =np.load('./data/p2_evaluation_reduced/y_test.npy') 
y_eval_train =np.load('./data/p2_evaluation_reduced/y_train.npy') 

X = np.load('./data/p2_unsupervised/X.npy')
X_tr = np.log2(X + 1)
X_eval_test =np.load('./data/p2_evaluation/X_test.npy') 
X_eval_test_tr = np.log2(X_eval_test + 1)
X_eval_train =np.load('./data/p2_evaluation/X_train.npy') 
X_eval_train_tr = np.log2(X_eval_train + 1)
y_eval_test =np.load('./data/p2_evaluation/y_test.npy') 
y_eval_train =np.load('./data/p2_evaluation/y_train.npy') 

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
mds.fit(Z[:, 0:1236])
plt.scatter(mds.embedding_[:,0], mds.embedding_[:,1])
plt.title("MDS Plot", size=18)
plt.axis("equal")
plt.show()

tsne = TSNE(n_components=2, perplexity=30)
z_tsne = tsne.fit_transform(Z[:, 0:1236])
plt.scatter(z_tsne[:, 0], z_tsne[:, 1])
plt.title("TSNE, perplexity=90", size=18)
plt.axis("equal")
plt.show()

kmeans = KMeans(n_clusters=3, n_init=100)
y = kmeans.fit_predict(Z)
plt.scatter(Z[:, 0], Z[:, 1], c=y)
plt.title("Scatter plot of 1st and 2nd Principal Components", size=10)
plt.xlabel("z1", size=14)
plt.ylabel("z2", size=14)
plt.axis("equal")
plt.show()

Z_cluster1 = Z[y==0]
Z_cluster2 = Z[y==1]
Z_cluster3 = Z[y==2]
pca1 = PCA()
pca2 = PCA()
pca3 = PCA()
Z1 = pca1.fit_transform(Z_cluster1)
Z2 = pca2.fit_transform(Z_cluster2)
Z3 = pca3.fit_transform(Z_cluster3)

plt.scatter(Z1[:, 0], Z1[:, 1])
plt.title("Scatter plot of 1st and 2nd Principal Components of Cluster 1", size=9)
plt.xlabel("z1", size=14)
plt.ylabel("z2", size=14)
plt.axis("equal")
plt.show()

plt.scatter(Z2[:, 0], Z2[:, 1])
plt.title("Scatter plot of 1st and 2nd Principal Components of Cluster 2", size=9)
plt.xlabel("z1", size=14)
plt.ylabel("z2", size=14)
plt.axis("equal")
plt.show()

plt.scatter(Z3[:, 0], Z3[:, 1])
plt.title("Scatter plot of 1st and 2nd Principal Components of Cluster 3", size=9)
plt.xlabel("z1", size=14)
plt.ylabel("z2", size=14)
plt.axis("equal")
plt.show()

kmeans = KMeans(n_clusters=3, n_init=100)
y = kmeans.fit_predict(Z)
y = kmeans.fit_predict(X_tr)
plt.scatter(Z[:, 0], Z[:, 1], c=y)
plt.title("Scatter plot of 1st and 2nd Principal Components", size=10)
plt.xlabel("z1", size=14)
plt.ylabel("z2", size=14)
plt.axis("equal")
plt.show()

y_train = y
X_train = X_tr



log_reg = LogisticRegressionCV(cv=5, Cs=[0.01, 0.1, 1, 10], penalty="l1", solver="liblinear", max_iter=5000, multi_class="ovr")
log_reg.fit(X_train, y_train)
print("score:", log_reg.score(X_train, y_train))
print("regularization C:", log_reg.C_)

arr = np.sum(np.abs(log_reg.coef_), axis = 0)
arr_max = np.max(np.abs(log_reg.coef_), axis = 0)
indices100 = arr.argsort()[-100:][::-1]
random100 = np.random.choice(X_tr.shape[1], 100, replace=False)
arr2 = np.var(X_tr, axis = 0)
topvar100 = arr2.argsort()[-100:][::-1]


indices70 = arr.argsort()[-70:][::-1]
random70 = np.random.choice(X_tr.shape[1], 70, replace=False)
topvar70 = arr2.argsort()[-70:][::-1]

X_train = X_eval_train_tr
y_train = y_eval_train
X_test = X_eval_test_tr
y_test = y_eval_test
log_reg2 = LogisticRegressionCV(cv=5, Cs=[0.01, 0.1, 1, 10], penalty="l1", solver="liblinear", max_iter=5000, multi_class="ovr")
log_reg2.fit(X_train[:, indices100], y_train)

log_reg3 = LogisticRegressionCV(cv=5, Cs=[0.01, 0.1, 1, 10], penalty="l1", solver="liblinear", max_iter=5000, multi_class="ovr")
log_reg3.fit(X_train[:, random100], y_train)

log_reg4 = LogisticRegressionCV(cv=5, Cs=[0.01, 0.1, 1, 10], penalty="l1", solver="liblinear", max_iter=5000, multi_class="ovr")
log_reg4.fit(X_train[:, topvar100], y_train)

print("Maximum score for top 100 features from LR:", np.round(log_reg2.score(X_test[:, indices100], y_test),2))
print("Maximum score for random 100 features:", np.round(log_reg3.score(X_test[:, random100], y_test),2))
print("Maximum score for 100 highest variance features:", np.round(log_reg4.score(X_test[:, topvar100], y_test),2))      

n_bins = 15
fig, axs = plt.subplots(1, 3, sharex = True, sharey=True, tight_layout=True)  
axs[0].hist(np.var(X_train[:, indices100], axis=0), bins=n_bins)
axs[1].hist(np.var(X_train[:, random100], axis=0), bins=n_bins)
axs[2].hist(np.var(X_train[:, topvar100], axis=0), bins=n_bins) 
#axs[3].hist(np.var(X_train[:, :], axis=0), bins=n_bins) 
axs[0].legend(["top 100 from LR"])
axs[1].legend(["100 random features"])
axs[2].legend(["top 100 highest variance"])

#axs[3].legend(["all features"])




tsne = TSNE(n_components=2, perplexity=40)
z_tsne = tsne.fit_transform(Z[:, 0:100])
plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y)
plt.title("TSNE, perplexity=30", size=18)
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

kmeans = KMeans(n_clusters=3, n_init=100)
y = kmeans.fit_predict(X_tr)
means = np.array([np.mean(X_tr[np.where(y==i)[0]],0) for i in range(5)])
mds_means = MDS(n_components=2)
mds_means.fit(means)

plt.scatter(Z[:, 0], Z[:, 1], c=y)
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


