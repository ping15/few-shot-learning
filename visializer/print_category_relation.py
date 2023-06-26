import random

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

from dataLoader.omniglotDataLoader import OminiglotDataLoader
from networks.relation import RelationNetwork
from utils import get_t_random_number, get_poly_random_number


def print_category(network, data_loader, batch_size):
    futures = []
    labels = []
    for i in range(batch_size):
        support_images, support_labels, query_images, query_labels = data_loader.sampleDataset()
        future = network.forward(support_images, support_labels, query_images, query_labels)
        futures.append(future)
        labels.append(query_labels)
    futures, labels = np.array(futures), np.array(labels)
    futures, labels = futures.reshape(([-1, 5])), labels.reshape([-1])

    pca = PCA(n_components=2)
    data = pca.fit_transform(futures)
    poly_points = get_poly_random_number(len(data))
    for i in range(len(data)):
        data[i][0] += poly_points[0][i]
        data[i][1] += poly_points[1][i]
        # data[i][1] += (random.random() * 0.2 - 0.1)

    colors = ['red', 'green', 'blue', 'yellow', 'black']
    fig, ax = plt.subplots()
    for i in range(len(labels)):
        ax.scatter(data[i, 0], data[i, 1], c=colors[labels[i]], s=10)
    ax.legend()
    plt.show()


encoder_weights_path = "../encoder_show.h5"
relation_weights_path = "../relation_show.h5"
data_loader = OminiglotDataLoader(basePath="../datas/omniglot_resized")

network = RelationNetwork(OminiglotDataLoader)
network.test(show=False)

print_category(network, data_loader, 10)
network.encoder.load_weights(encoder_weights_path)
network.relationModel.load_weights(relation_weights_path)
print_category(network, data_loader, 10)
