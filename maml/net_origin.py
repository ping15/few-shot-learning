import pickle
import random

import tensorflow as tf

from dataLoader.omniglotDataLoader import OmniglotDataLoader
from config import args


class MAML:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.meta_model = self.get_maml_model()
        self.data_loader = OmniglotDataLoader()
        self.batch_size = args.batch_size

    def get_batch_data(self, training=True):
        return self.data_loader.sample_batch_dataset(self.batch_size, training=training)

    def get_maml_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=1,
                                   input_shape=self.input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.num_classes, activation='softmax'),
        ])

        return model

    def shuffle(self, images, labels):
        indexs = list(range(len(images)))
        random.shuffle(indexs)
        shuffle_images = tf.gather(images, indexs)
        shuffle_labels = tf.gather(labels, indexs)
        return shuffle_images, shuffle_labels

    def train_on_batch(self, train_data, inner_optimizer, inner_step,
                       outer_optimizer=None, train_step=True, training=True):
        batch_acc = []
        batch_loss = []
        task_weights = []

        meta_weights = self.meta_model.get_weights()

        meta_support_image, meta_support_label, meta_query_image, meta_query_label = next(train_data)
        for support_image, support_label in zip(meta_support_image, meta_support_label):
            self.meta_model.set_weights(meta_weights)
            for _ in range(inner_step):
                support_image, support_label = self.shuffle(support_image, support_label)
                with tf.GradientTape() as tape:
                    logits = self.meta_model(support_image, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(support_label, logits)
                    loss = tf.reduce_mean(loss)

                grads = tape.gradient(loss, self.meta_model.trainable_variables)
                inner_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

            task_weights.append(self.meta_model.get_weights())

        with tf.GradientTape() as tape:
            for i, (query_image, query_label) in enumerate(zip(meta_query_image, meta_query_label)):
                self.meta_model.set_weights(task_weights[i])
                with tf.GradientTape() as t:
                    logits = self.meta_model(query_image, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(query_label, logits)

                gradients = t.gradient(loss, self.meta_model.trainable_variables)
                inner_optimizer.apply_gradients(zip(gradients, self.meta_model.trainable_variables))

                loss = tf.reduce_mean(loss)
                batch_loss.append(loss)

                acc = tf.cast(tf.argmax(logits, axis=-1) == query_label, tf.float32)
                acc = tf.reduce_mean(acc)
                batch_acc.append(acc)

                # with open("matching_omniglot_5way_1shot_label_prediction.pkl", "wb") as f:
                #     pickle.dump((tf.one_hot(query_label, axis=-1, depth=5), logits), f)

            mean_acc = tf.reduce_mean(batch_acc)
            mean_loss = tf.reduce_mean(batch_loss)

        self.meta_model.set_weights(meta_weights)
        if outer_optimizer:
            grads = tape.gradient(mean_loss, self.meta_model.trainable_variables)
            outer_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

        return mean_loss, mean_acc
