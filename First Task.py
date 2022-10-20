from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

X = np.load('activations/activations_0_both.npy')
arr = np.array([6, 1, 625, 50])
arr2 = arr[2:]
print (arr2)


"""rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal');
pca = PCA(n_components=2)
pca.fit(X)
PCA(n_components=2)
print(pca.components_)
print(pca.explained_variance_)

pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)

X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal');"""