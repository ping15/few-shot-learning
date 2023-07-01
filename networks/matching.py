import pickle
import time

import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

from models.matching import CNNEncoder, GEmbeddingBidirectionalLSTM, FEmbeddingBidirectionalLSTM, DistanceNetwork
from configs import settings
from .base import BaseNetwork


class MatchingNetwork(BaseNetwork):
    def __init__(self, data_loader=None, use_embedding=True):
        super(MatchingNetwork, self).__init__(data_loader)
        self.use_embedding = use_embedding
        self.encoder = CNNEncoder()
        self.g_embedding = GEmbeddingBidirectionalLSTM(settings.LAYER_SIZES, settings.BATCH_SIZE)
        self.f_embedding = FEmbeddingBidirectionalLSTM(settings.UNITS)
        self.cos_distance = DistanceNetwork(settings.TRAIN_TEST_WAY)

    def forward(self, train_images, train_labels, test_images, test_labels):
        """
        :param train_images: [batch_size, class_num, num_per_class,
                              image_height, image_width, image_channel]
        :param train_labels: [batch_size, class_num, num_per_class]
        :param test_images: [batch_szie, class_num, query_num,
                             image_height, image_width, image_channel]
        :param test_labels: [batch_size, class_num, query_num]
        :return:
        """
        train_futures = self.encoder(train_images, training=True)
        test_futures = self.encoder(test_images, training=True)

        if self.use_embedding:
            train_image_embeddings = self.g_embedding(train_futures, training=True)
            test_image_embeddings = self.f_embedding(train_image_embeddings, test_futures, training=True)
        else:
            train_image_embeddings = train_futures
            test_image_embeddings = test_futures

        result = self.cos_distance(train_image_embeddings, train_labels, test_image_embeddings)

        result = tf.keras.layers.Softmax(axis=-1)(result)

        return result

    @property
    def trainable_variables(self):
        if self.use_embedding:
            return self.encoder.trainable_variables + self.g_embedding.trainable_variables + \
                   self.f_embedding.trainable_variables
        return self.encoder.trainable_variables

    @property
    def loss_function(self):
        return tf.keras.losses.MeanSquaredError()
        # return tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def train(self, epochs, count_per_epoch):
        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []
        predictions_list = []
        labels_list = []
        total_time = 0
        for epoch in range(epochs):
            start = time.time()
            self.train_loss.reset_state()
            self.train_accuracy.reset_state()
            self.test_loss.reset_state()
            self.test_accuracy.reset_state()

            for _ in range(count_per_epoch):
                train_images, train_labels, test_images, test_labels = \
                    self.dataLoader.sample_batch_dataset(settings.BATCH_SIZE, training=True)
                self.train_step(train_images, train_labels, test_images, test_labels)

            for _ in range(count_per_epoch // 2):
                train_images, train_labels, test_images, test_labels = \
                    self.dataLoader.sample_batch_dataset(settings.BATCH_SIZE, training=False)
                oneHotLabels, predictions = self.test_step(train_images, train_labels, test_images, test_labels)
                predictions_list.append(predictions)
                labels_list.append(oneHotLabels)

            train_loss.append(self.train_loss.result())
            train_accuracy.append(self.train_accuracy.result())
            test_loss.append(self.test_loss.result())
            test_accuracy.append(self.test_accuracy.result())
            print(
                "Epoch: {:.2f} ".format(epoch + 1),
                "train_loss: {:.2f} ".format(self.train_loss.result()),
                "train_accuracy: {:.2f}% ".format(self.train_accuracy.result() * 100),
                "test_loss: {:.2f} ".format(self.test_loss.result()),
                "test_accuracy: {:.2f}% ".format(self.test_accuracy.result() * 100),
                "time: {:.2f} ".format(time.time() - start)
            )
            total_time = total_time + time.time() - start
            if (epoch + 1) % 5 == 0:
                print("time: {:.2f} ".format(total_time * 20 / 3600))
                total_time = 0
        with open("maml_omniglot_5way_1shot.pkl", "wb") as f:
            pickle.dump((train_loss, train_accuracy, test_loss, test_accuracy), f)

        with open("maml_omniglot_5way_1shot_label_prediction.pkl", "wb") as f:
            pickle.dump((labels_list, predictions_list), f)

    def train_step(self, train_images, train_labels, test_images, test_labels):
        with tf.GradientTape() as tape:
            predictions = self.forward(train_images, train_labels, test_images, test_labels)
            one_hot_labels = self.labelEncode(test_labels)
            # print(predictions.shape, one_hot_labels.shape)

            loss = self.loss_function(one_hot_labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(one_hot_labels, predictions)

    def test_step(self, train_images, train_labels, test_images, test_labels):
        predictions = self.forward(train_images, train_labels, test_images, test_labels)
        one_hot_labels = self.labelEncode(test_labels)
        loss = self.loss_function(one_hot_labels, predictions)

        self.test_loss(loss)
        self.test_accuracy(one_hot_labels, predictions)
        return one_hot_labels, predictions
