import random

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from dataLoader.omniglotDataLoader import OminiglotDataLoader
from maml.net import MAML
from maml.dataReader import MAMLDataLoader
from maml.config import args


def print_category(network, data_loader, batch_size):
    futures = []
    labels = []
    for i in range(batch_size):
        support_images, support_labels, query_images, query_labels = data_loader.sampleDataset()
        future = network(support_images, support_labels, query_images, query_labels)
        futures.append(future)
        labels.append(query_labels)
    futures, labels = np.array(futures), np.array(labels)
    futures, labels = futures.reshape(([-1, 5])), labels.reshape([-1])

    pca = PCA(n_components=2)
    data = pca.fit_transform(futures)
    for i in range(len(data)):
        data[i][0] += (random.random() * 0.2 - 0.1)
        data[i][1] += (random.random() * 0.2 - 0.1)

    colors = ['red', 'green', 'blue', 'yellow', 'black']
    fig, ax = plt.subplots()
    for i in range(len(labels)):
        ax.scatter(data[i, 0], data[i, 1], c=colors[labels[i]], s=10)
    ax.legend()
    plt.show()


maml_weights_path = "../maml/maml2_5way_5shot.h5"
# data_loader = OminiglotDataLoader(basePath="../datas/omniglot_resized")
train_data = MAMLDataLoader(args.train_data_dir, args.batch_size)
val_data = MAMLDataLoader(args.val_data_dir, args.val_batch_size)

inner_optimizer = tf.keras.optimizers.Adam(args.inner_lr)
outer_optimizer = tf.keras.optimizers.Adam(args.outer_lr)


network = MAML(input_shape=(28, 28, 1), num_classes=5)

print_category(network.meta_model, val_data, 1)
network.meta_model.load_weights(maml_weights_path)
print_category(network.meta_model, val_data, 1)
