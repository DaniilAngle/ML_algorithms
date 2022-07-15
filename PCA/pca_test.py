import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from PCA.pca import PCA

data = datasets.load_iris()
X = data.data
y = data.target

pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(x1, x2,
            c=y, edgecolors="none", alpha=0.8,
            cmap=plt.get_cmap('viridis', 3))

plt.xlabel('Princ component 1')
plt.ylabel('Princ component 2')
plt.colorbar()
plt.show()