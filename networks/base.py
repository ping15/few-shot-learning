import pickle
import time

import tensorflow as tf

from configs import settings
from dataLoader.utils import showTest


class BaseNetwork(object):
    def __init__(self, data_loader):
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy",)
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")

        self.dataLoader = data_loader()

        # self.lossFn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()

    def forward(self, train_images, train_labels, test_images, test_labels):
        raise NotImplementedError("According support_set and query_set to generate predictions")

    def test(self, title=None, fig_text=None, show=True):
        train_images, train_labels, test_images, test_labels = self.dataLoader.sample_dataset_from_test()
        predictions = tf.argmax(self.forward(train_images, train_labels, test_images, test_labels), axis=-1)
        if show:
            showTest(test_images, test_labels, predictions, title, fig_text)

    def labelEncode(self, labels):
        # return tf.one_hot(labels, depth=settings.TRAIN_TEST_WAY, axis=1)
        return tf.one_hot(labels, depth=settings.TRAIN_TEST_WAY, axis=-1)

    @property
    def trainable_variables(self):
        return NotImplementedError("Return model trainable variables")

    @property
    def loss_function(self):
        return NotImplementedError("Loss function to compute predictions and labels")

    def train_step(self, train_images, train_labels, test_images, test_labels):
        with tf.GradientTape() as tape:
            predictions = self.forward(train_images, train_labels, test_images, test_labels)
            oneHotLabels = self.labelEncode(test_labels)
            # print(oneHotLabels.shape, predictions.shape)
            loss = self.loss_function(oneHotLabels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(oneHotLabels, predictions)

    def test_step(self, train_images, train_labels, test_images, test_labels):
        predictions = self.forward(train_images, train_labels, test_images, test_labels)
        oneHotLabels = self.labelEncode(test_labels)
        loss = self.loss_function(oneHotLabels, predictions)

        self.test_loss(loss)
        self.test_accuracy(oneHotLabels, predictions)
        return oneHotLabels, predictions

    def train(self, epochs, count_per_epoch):
        # self.test()
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
                train_images, train_labels, test_images, test_labels = self.dataLoader.get_dataset()
                # train_images, train_labels, test_images, test_labels = self.dataLoader.sampleBatchDataset(
                #     batch_size=settings.BATCH_SIZE, training=True, resize=True
                # )
                self.train_step(train_images, train_labels, test_images, test_labels)

            for _ in range(count_per_epoch // 2):
                train_images, train_labels, test_images, test_labels = self.dataLoader.sample_dataset_from_test()
                # train_images, train_labels, test_images, test_labels = self.dataLoader.sampleBatchDataset(
                #     batch_size=settings.BATCH_SIZE, training=False, resize=True
                # )
                oneHotLabels, predictions = self.test_step(train_images, train_labels, test_images, test_labels)
                # print(predictions.shape)
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
        # self.test()
