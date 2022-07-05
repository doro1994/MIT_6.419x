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


X = np.load('./data/p1/X.npy')
X_tr = np.log2(X + 1)

#pca = PCA(n_components=100)
#pca.fit(X)
#components = (np.cumsum(pca.explained_variance_ratio_) < 0.85).astype('int')

#print(sum(components)+1)
pca = PCA()
Z = pca.fit_transform(X_tr)
components = (np.cumsum(pca.explained_variance_ratio_) < 0.85).astype('int')
plt.scatter(Z[:, 0], Z[:, 1])
plt.title("Scatter plot of 1st and 2nd Principal Components", size=10)
plt.xlabel("z1", size=14)
plt.ylabel("z2", size=14)
plt.axis("equal")
plt.show()


tsne10 = TSNE(n_components=2, perplexity=40)
tsne50 = TSNE(n_components=2, perplexity=40)
tsne100 = TSNE(n_components=2, perplexity=40)
tsne250 = TSNE(n_components=2, perplexity=40)
tsne500 = TSNE(n_components=2, perplexity=40)
z_tsne10 = tsne10.fit_transform(Z[:, 0:10])
z_tsne50 = tsne50.fit_transform(Z[:, 0:50])
z_tsne100 = tsne100.fit_transform(Z[:, 0:100])
z_tsne250 = tsne250.fit_transform(Z[:, 0:250])
z_tsne500 = tsne500.fit_transform(Z[:, 0:500])

fig, axs = plt.subplots(2, 3, tight_layout=True)
axs[0,0].scatter(z_tsne10[:, 0], z_tsne10[:, 1])
axs[0,1].scatter(z_tsne50[:, 0], z_tsne50[:, 1])
axs[0,2].scatter(z_tsne100[:, 0], z_tsne100[:, 1])
axs[1,0].scatter(z_tsne250[:, 0], z_tsne250[:, 1])
axs[1,1].scatter(z_tsne500[:, 0], z_tsne500[:, 1])

axs[0,0].set_title("10 PC")
axs[0,1].set_title("50 PC")
axs[0,2].set_title("100 PC")
axs[1,0].set_title("250 PC")
axs[1,1].set_title("500 PC")

""" CATEGORY A (visualization) """
""" tSNE perplexity test """

tsne2 = TSNE(n_components=2, perplexity=2)
tsne5 = TSNE(n_components=2, perplexity=5)
tsne10 = TSNE(n_components=2, perplexity=10)
tsne30 = TSNE(n_components=2, perplexity=30)
tsne50 = TSNE(n_components=2, perplexity=50)
tsne100 = TSNE(n_components=2, perplexity=100)
z_tsne2 = tsne2.fit_transform(Z[:, 0:50])
z_tsne5 = tsne5.fit_transform(Z[:, 0:50])
z_tsne10 = tsne10.fit_transform(Z[:, 0:50])
z_tsne30 = tsne30.fit_transform(Z[:, 0:50])
z_tsne50 = tsne50.fit_transform(Z[:, 0:50])
z_tsne100 = tsne100.fit_transform(Z[:, 0:50])

fig, axs = plt.subplots(2, 3, tight_layout=True)
axs[0,0].scatter(z_tsne2[:, 0], z_tsne2[:, 1])
axs[0,1].scatter(z_tsne5[:, 0], z_tsne5[:, 1])
axs[0,2].scatter(z_tsne10[:, 0], z_tsne10[:, 1])
axs[1,0].scatter(z_tsne30[:, 0], z_tsne30[:, 1])
axs[1,1].scatter(z_tsne50[:, 0], z_tsne50[:, 1])
axs[1,2].scatter(z_tsne100[:, 0], z_tsne100[:, 1])

axs[0,0].set_title("Perplexity 2")
axs[0,1].set_title("Perplexity 5")
axs[0,2].set_title("Perplexity 10")
axs[1,0].set_title("Perplexity 30")
axs[1,1].set_title("Perplexity 50")
axs[1,2].set_title("Perplexity 100")


""" tSNE learning rate test """

tsne10 = TSNE(n_components=2, perplexity=40, learning_rate = 10)
tsne50 = TSNE(n_components=2, perplexity=40, learning_rate = 50)
tsne100 = TSNE(n_components=2, perplexity=40, learning_rate = 100)
tsne250 = TSNE(n_components=2, perplexity=40, learning_rate = 250)
tsne500 = TSNE(n_components=2, perplexity=40, learning_rate = 500)
tsne1000 = TSNE(n_components=2, perplexity=40, learning_rate = 1000)
z_tsne10 = tsne10.fit_transform(Z[:, 0:50])
z_tsne50 = tsne50.fit_transform(Z[:, 0:50])
z_tsne100 = tsne100.fit_transform(Z[:, 0:50])
z_tsne250 = tsne250.fit_transform(Z[:, 0:50])
z_tsne500 = tsne500.fit_transform(Z[:, 0:50])
z_tsne1000 = tsne1000.fit_transform(Z[:, 0:50])

fig, axs = plt.subplots(2, 3, tight_layout=True)
axs[0,0].scatter(z_tsne10[:, 0], z_tsne10[:, 1])
axs[0,1].scatter(z_tsne50[:, 0], z_tsne50[:, 1])
axs[0,2].scatter(z_tsne100[:, 0], z_tsne100[:, 1])
axs[1,0].scatter(z_tsne250[:, 0], z_tsne250[:, 1])
axs[1,1].scatter(z_tsne500[:, 0], z_tsne500[:, 1])
axs[1,2].scatter(z_tsne1000[:, 0], z_tsne1000[:, 1])

axs[0,0].set_title("Learning rate 10")
axs[0,1].set_title("Learning rate 50")
axs[0,2].set_title("Learning rate 100")
axs[1,0].set_title("Learning rate 250")
axs[1,1].set_title("Learning rate 500")
axs[1,2].set_title("Learning rate 1000")

""" CATEGORY B (clustering/feature selection) """
""" Effect of number of PC chosen on clustering """

kmeans = KMeans(n_clusters=3, n_init=100)
all_kmeans = [KMeans(n_clusters=3, n_init=100) for _ in range(8)]
y2 = all_kmeans[0].fit_predict(Z[:, :2])
y10 = all_kmeans[1].fit_predict(Z[:, :10])
y50 = all_kmeans[2].fit_predict(Z[:, :50])
y100 = all_kmeans[3].fit_predict(Z[:, :100])
y500 = all_kmeans[4].fit_predict(Z[:, :500])
y_all = all_kmeans[5].fit_predict(Z)

fig, axs = plt.subplots(2, 3, tight_layout=True)
axs[0,0].scatter(Z[:, 0], Z[:, 1], c=y2)
axs[0,1].scatter(Z[:, 0], Z[:, 1], c=y10)
axs[0,2].scatter(Z[:, 0], Z[:, 1], c=y50)
axs[1,0].scatter(Z[:, 0], Z[:, 1], c=y100)
axs[1,1].scatter(Z[:, 0], Z[:, 1], c=y500)
axs[1,2].scatter(Z[:, 0], Z[:, 1], c=y_all)

axs[0,0].set_title("kmeans for 2 PC")
axs[0,1].set_title("kmeans for 10 PC")
axs[0,2].set_title("kmeans for 50 PC")
axs[1,0].set_title("kmeans for 100 PC")
axs[1,1].set_title("kmeans for 500 PC")
axs[1,2].set_title("kmeans for all PC")


""" Magnitude of reulariazation """

X = np.load('./data/p2_unsupervised_reduced/X.npy')
X_tr = np.log2(X + 1)
X_eval_test =np.load('./data/p2_evaluation_reduced/X_test.npy') 
X_eval_test_tr = np.log2(X_eval_test + 1)
X_eval_train =np.load('./data/p2_evaluation_reduced/X_train.npy') 
X_eval_train_tr = np.log2(X_eval_train + 1)
y_eval_test =np.load('./data/p2_evaluation_reduced/y_test.npy') 
y_eval_train =np.load('./data/p2_evaluation_reduced/y_train.npy') 


pca = PCA()
Z = pca.fit_transform(X_tr)

kmeans = KMeans(n_clusters=3, n_init=100)
y = kmeans.fit_predict(Z)

y_train = y
X_train = X_tr


log_reg0001 = LogisticRegression(C=0.001, penalty="l1", solver="liblinear", max_iter=5000, multi_class="ovr")
log_reg001 = LogisticRegression(C=0.01, penalty="l1", solver="liblinear", max_iter=5000, multi_class="ovr")
log_reg01 = LogisticRegression(C=0.1, penalty="l1", solver="liblinear", max_iter=5000, multi_class="ovr")
log_reg1 = LogisticRegression(C=1, penalty="l1", solver="liblinear", max_iter=5000, multi_class="ovr")
log_reg10 = LogisticRegression(C=10, penalty="l1", solver="liblinear", max_iter=5000, multi_class="ovr")
log_reg100 = LogisticRegression(C=100, penalty="l1", solver="liblinear", max_iter=5000, multi_class="ovr")

log_reg0001.fit(X_train, y_train)
log_reg001.fit(X_train, y_train)
log_reg01.fit(X_train, y_train)
log_reg1.fit(X_train, y_train)
log_reg10.fit(X_train, y_train)
log_reg100.fit(X_train, y_train)

print("score C=0.001:", log_reg0001.score(X_train, y_train))
print("score C=0.01:", log_reg001.score(X_train, y_train))
print("score C=0.1:", log_reg01.score(X_train, y_train))
print("score C=1:", log_reg1.score(X_train, y_train))
print("score C=10:", log_reg10.score(X_train, y_train))
print("score C=100:", log_reg100.score(X_train, y_train))

#print("regularization C:", log_reg.C_)

arr0001 = np.sum(np.abs(log_reg0001.coef_), axis = 0)
arr001 = np.sum(np.abs(log_reg001.coef_), axis = 0)
arr01 = np.sum(np.abs(log_reg01.coef_), axis = 0)
arr1 = np.sum(np.abs(log_reg1.coef_), axis = 0)
arr10 = np.sum(np.abs(log_reg10.coef_), axis = 0)
arr100 = np.sum(np.abs(log_reg100.coef_), axis = 0)

indices0001 = arr0001.argsort()[-100:][::-1]
indices001 = arr001.argsort()[-100:][::-1]
indices01 = arr01.argsort()[-100:][::-1]
indices1 = arr1.argsort()[-100:][::-1]
indices10 = arr10.argsort()[-100:][::-1]
indices100 = arr100.argsort()[-100:][::-1]

X_train = X_eval_train_tr
y_train = y_eval_train
X_test = X_eval_test_tr
y_test = y_eval_test

log_reg_eval_0001 = LogisticRegressionCV(cv=7, Cs=[0.001, 0.01, 0.1, 1, 10, 100], penalty="l1", solver="liblinear", max_iter=5000, multi_class="ovr")
log_reg_eval_001 = LogisticRegressionCV(cv=7, Cs=[0.001, 0.01, 0.1, 1, 10, 100], penalty="l1", solver="liblinear", max_iter=5000, multi_class="ovr")
log_reg_eval_01 = LogisticRegressionCV(cv=7, Cs=[0.001, 0.01, 0.1, 1, 10, 100], penalty="l1", solver="liblinear", max_iter=5000, multi_class="ovr")
log_reg_eval_1 = LogisticRegressionCV(cv=7, Cs=[0.001, 0.01, 0.1, 1, 10, 100], penalty="l1", solver="liblinear", max_iter=5000, multi_class="ovr")
log_reg_eval_10 = LogisticRegressionCV(cv=7, Cs=[0.001, 0.01, 0.1, 1, 10, 100], penalty="l1", solver="liblinear", max_iter=5000, multi_class="ovr")
log_reg_eval_100 = LogisticRegressionCV(cv=7, Cs=[0.001, 0.01, 0.1, 1, 10, 100], penalty="l1", solver="liblinear", max_iter=5000, multi_class="ovr")

log_reg_eval_0001.fit(X_train[:, indices0001], y_train)
log_reg_eval_001.fit(X_train[:, indices001], y_train)
log_reg_eval_01.fit(X_train[:, indices01], y_train)
log_reg_eval_1.fit(X_train[:, indices1], y_train)
log_reg_eval_10.fit(X_train[:, indices10], y_train)
log_reg_eval_100.fit(X_train[:, indices100], y_train)

print("Score for 100 features with C=0.001 LR:", np.round(log_reg_eval_0001.score(X_test[:, indices0001], y_test),2))
print("Score for 100 features with C=0.01 LR:", np.round(log_reg_eval_001.score(X_test[:, indices001], y_test),2))
print("Score for 100 features with C=0.1 LR:", np.round(log_reg_eval_01.score(X_test[:, indices01], y_test),2))
print("Score for 100 features with C=1 LR:", np.round(log_reg_eval_1.score(X_test[:, indices1], y_test),2))
print("Score for 100 features with C=10 LR:", np.round(log_reg_eval_10.score(X_test[:, indices10], y_test),2))
print("Score for 100 features with C=100 LR:", np.round(log_reg_eval_100.score(X_test[:, indices100], y_test),2))

""" Type of regularization """

pca = PCA()
Z = pca.fit_transform(X_tr)

kmeans = KMeans(n_clusters=3, n_init=100)
y = kmeans.fit_predict(Z)

y_train = y
X_train = X_tr

log_reg_L1 = LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=5000, multi_class="ovr")
log_reg_L2 = LogisticRegression(C=0.1, penalty="l2", solver="saga", max_iter=5000, multi_class="ovr")
log_reg_EL = LogisticRegression(C=0.1, penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=5000, multi_class="ovr")

log_reg_L1.fit(X_train, y_train)
log_reg_L2.fit(X_train, y_train)
log_reg_EL.fit(X_train, y_train)

print("score for L1 regularization:", log_reg_L1.score(X_train, y_train))
print("score for L2 regularization:", log_reg_L2.score(X_train, y_train))
print("score for Elastinet regularization:", log_reg_EL.score(X_train, y_train))

#print("regularization C:", log_reg.C_)

arrL1 = np.sum(np.abs(log_reg_L1.coef_), axis = 0)
arrL2 = np.sum(np.abs(log_reg_L2.coef_), axis = 0)
arrEL = np.sum(np.abs(log_reg_EL.coef_), axis = 0)

indicesL1 = arrL1.argsort()[-100:][::-1]
indicesL2 = arrL2.argsort()[-100:][::-1]
indicesEL = arrEL.argsort()[-100:][::-1]

X_train = X_eval_train_tr
y_train = y_eval_train
X_test = X_eval_test_tr
y_test = y_eval_test

log_reg_eval_L1_L1 = LogisticRegression(C=0.01, penalty="l1", solver="saga", max_iter=5000, multi_class="ovr")
log_reg_eval_L2_L1 = LogisticRegression(C=0.01, penalty="l1", solver="saga", max_iter=5000, multi_class="ovr")
log_reg_eval_EL_L1 = LogisticRegression(C=0.01, penalty="l1", solver="saga", max_iter=5000, multi_class="ovr")
log_reg_eval_L1_L2 = LogisticRegression(C=0.01, penalty="l2", solver="saga", max_iter=5000, multi_class="ovr")
log_reg_eval_L2_L2 = LogisticRegression(C=0.01, penalty="l2", solver="saga", max_iter=5000, multi_class="ovr")
log_reg_eval_EL_L2 = LogisticRegression(C=0.01, penalty="l2", solver="saga", max_iter=5000, multi_class="ovr")
log_reg_eval_L1_EL = LogisticRegression(C=0.01, penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=5000, multi_class="ovr")
log_reg_eval_L2_EL = LogisticRegression(C=0.01, penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=5000, multi_class="ovr")
log_reg_eval_EL_EL = LogisticRegression(C=0.01, penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=5000, multi_class="ovr")

log_reg_eval_L1_L1.fit(X_train[:, indicesL1], y_train)
log_reg_eval_L2_L1.fit(X_train[:, indicesL2], y_train)
log_reg_eval_EL_L1.fit(X_train[:, indicesEL], y_train)
log_reg_eval_L1_L2.fit(X_train[:, indicesL1], y_train)
log_reg_eval_L2_L2.fit(X_train[:, indicesL2], y_train)
log_reg_eval_EL_L2.fit(X_train[:, indicesEL], y_train)
log_reg_eval_L1_EL.fit(X_train[:, indicesL1], y_train)
log_reg_eval_L2-EL.fit(X_train[:, indicesL2], y_train)
log_reg_eval_EL-EL.fit(X_train[:, indicesEL], y_train)

print("Score for 100 features from LR with penalty L1 evaluated on LR with penalty L1:", 
      np.round(log_reg_eval_L1_L1.score(X_test[:, indicesL1], y_test),2))
print("Score for 100 features from LR with penalty L2 evaluated on LR with penalty L1:", 
      np.round(log_reg_eval_L2_L1.score(X_test[:, indicesL2], y_test),2))
print("Score for 100 features from LR with penalty 'elasticnet' evaluated on LR with penalty L1:", 
      np.round(log_reg_eval_EL_L1.score(X_test[:, indicesEL], y_test),2))
print("Score for 100 features from LR with penalty L1 evaluated on LR with penalty L2:", 
      np.round(log_reg_eval_L1_L2.score(X_test[:, indicesL1], y_test),2))
print("Score for 100 features from LR with penalty L2 evaluated on LR with penalty L2:", 
      np.round(log_reg_eval_L2_L2.score(X_test[:, indicesL2], y_test),2))
print("Score for 100 features from LR with penalty 'elasticnet' evaluated on LR with penalty L2:", 
      np.round(log_reg_eval_EL_L2.score(X_test[:, indicesEL], y_test),2))
print("Score for 100 features from LR with penalty L1 evaluated on LR with penalty 'elasticnet':", 
      np.round(log_reg_eval_L1_EL.score(X_test[:, indicesL1], y_test),2))
print("Score for 100 features from LR with penalty L2 evaluated on LR with penalty 'elasticnet':", 
      np.round(log_reg_eval_L2_EL.score(X_test[:, indicesL2], y_test),2))
print("Score for 100 features from LR with penalty 'elasticnet' evaluated on LR with penalty 'elasticnet':", 
      np.round(log_reg_eval_EL_EL.score(X_test[:, indicesEL], y_test),2))

