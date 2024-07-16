import numpy as np
from sklearn import datasets
from sklearn.manifold import TSNE
import os
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_tsne3d(features, labels, epoch, fileNameDir=None, num_classes=5):
    if not os.path.exists(fileNameDir):
        os.makedirs(fileNameDir)

    tsne = TSNE(n_components=3, init='pca', random_state=0)
    tsne_features = tsne.fit_transform(features)

    scatter_s = 35
    palette = sns.color_palette("pastel", n_colors=num_classes)
    hex = palette.as_hex()

    fig = plt.figure()
    ax = Axes3D(fig)

    for x, y, z, v in zip(tsne_features[:, 0], tsne_features[:, 1], tsne_features[:, 2], labels):
        ax.scatter(x, y, z, c=hex[int(v)], marker=".", s=scatter_s)

    plt.savefig(os.path.join(fileNameDir, f"{epoch}.png"), dpi=300)
    plt.close()


if __name__ == '__main__':
    digits = datasets.load_digits(n_class=5)
    features, labels = digits.data, digits.target
    file_name = "test"
    file_dir = 'result/test'
    plot_tsne3d(features=features, labels=labels, epoch=file_name, fileNameDir=file_dir, num_classes=5)
