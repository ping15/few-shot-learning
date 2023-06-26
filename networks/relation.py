import tensorflow as tf

from models import RelationModel, CNNEncoder
from configs import settings
from .base import BaseNetwork


class RelationNetwork(BaseNetwork):
    def __init__(self, data_loader=None, weights=None):
        super(RelationNetwork, self).__init__(data_loader)
        self.encoder = CNNEncoder()
        self.relationModel = RelationModel(settings.RELATION_DIM)

    def forward(self, train_images, train_labels, test_images, test_labels, visible=False):
        train_futures = self.encoder(train_images, training=True)
        test_futures = self.encoder(test_images, training=True)
        train_futures = tf.repeat(tf.expand_dims(train_futures, axis=0),
                                  settings.TEST_SHOT * settings.TRAIN_TEST_WAY, axis=0)
        test_futures = tf.repeat(tf.expand_dims(test_futures, axis=1),
                                 settings.TRAIN_SHOT * settings.TRAIN_TEST_WAY, axis=1)
        concat_futures = tf.concat([train_futures, test_futures], 4)

        shapes = concat_futures.shape
        concat_futures = tf.reshape(concat_futures, [
            -1,
            shapes[2],
            shapes[3],
            settings.FEATURE_DIM * 2])

        relations = self.relationModel(concat_futures, training=True)
        relations = tf.reshape(relations, [-1, settings.TRAIN_TEST_WAY])

        return relations

    @property
    def trainable_variables(self):
        return self.encoder.trainable_variables + self.relationModel.trainable_variables

    @property
    def loss_function(self):
        return tf.keras.losses.MeanSquaredError()
        # return tf.keras.losses.CategoricalCrossentropy()

    def train(self, epochs, count_per_epoch):
        self.test(show=False)
        # self.encoder.load_weights("encoder_show.h5")
        # self.relationModel.load_weights("relation_show.h5")
        super(RelationNetwork, self).train(epochs, count_per_epoch)
        self.encoder.save_weights("encoder_omniglot_5way_1shot.h5")
        self.relationModel.save_weights("relation_omniglot_5way_1shot.h5")
        # self.test()
