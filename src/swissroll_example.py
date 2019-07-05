import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#  import sys
#  sys.path.append("../src/")
from sklearn import datasets
from mind import mind_ensemble

X, color = datasets.samples_generator.make_swiss_roll(n_samples=200)
# Sort data by position on manifold
X_sort = X[np.argsort(color)]

case=1

if case==0:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_sort[:,0], X_sort[:,1],X_sort[:,2], c=np.arange(X.shape[0]))
    plt.show()

elif case==1:
    m = mind_ensemble(X_sort, manifold_dim=2, n_trees=100, seed=123)
    m.learn_coordinates()
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(m.P)
    ax[1].imshow(m.D)

    ax[0].set_title("Transition probabilities");
    ax[1].set_title("Pairwise distances");

    plt.figure()
    plt.scatter(m.y[:,0], m.y[:,1], c=np.arange(X.shape[0]))
    plt.xlabel("Manifold Dim. 1")
    plt.ylabel("Manifold Dim. 2")

    plt.show()
