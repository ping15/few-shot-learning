import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

from models import Encoder
from .base import BaseNetwork
from configs import settings

#     def loss(self, sample):
#         xs = Variable(sample['xs']) # support
#         xq = Variable(sample['xq']) # query
#
#         n_class = xs.size(0)
#         assert xq.size(0) == n_class
#         n_support = xs.size(1)
#         n_query = xq.size(1)
#
#         target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
#         target_inds = Variable(target_inds, requires_grad=False)
#
#         if xq.is_cuda:
#             target_inds = target_inds.cuda()
#
#         x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
#                        xq.view(n_class * n_query, *xq.size()[2:])], 0)
#
#         z = self.encoder.forward(x)
#         z_dim = z.size(-1)
#
#         z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
#         zq = z[n_class*n_support:]
#
#         dists = euclidean_dist(zq, z_proto)
#
#         log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
#
#         loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
#
#         _, y_hat = log_p_y.max(2)
#         acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
#
#         return loss_val, {
#             'loss': loss_val.item(),
#             'acc': acc_val.item()
#         }


class PrototypicalNetwork(BaseNetwork):
    def __init__(self, dataLoader):
        super(PrototypicalNetwork, self).__init__(dataLoader)
        self.encoder = Encoder()

    def euclidean_dist(self, features1, features2):
        m, n = features1.shape[0], features2.shape[0]

        features1 = tf.repeat(tf.reshape(features1, [m, 1, -1]), n, axis=1)
        features2 = tf.repeat(tf.reshape(features2, [1, n, -1]), m, axis=0)

        return tf.reduce_sum(tf.square(features1 - features2), axis=2)

    def get_prototypes(self, images):
        return tf.reduce_mean(images, axis=1)

    def forward(self, train_images, train_labels, test_images, test_labels):
        concat_images = tf.concat([train_images, test_images], axis=0)

        features = self.encoder(concat_images, training=True)
        train_features, test_features = features[:settings.TRAIN_SHOT * settings.TRAIN_TEST_WAY], \
                                        features[settings.TRAIN_SHOT * settings.TRAIN_TEST_WAY:]

        train_features = tf.reshape(train_features, [settings.TRAIN_TEST_WAY, settings.TRAIN_SHOT, -1])
        prototypes = self.get_prototypes(train_features)

        distances = self.euclidean_dist(test_features, prototypes)

        # predictions = tf.math.log(tf.nn.softmax(-distances, axis=1)), [
        #     settings.TRAIN_TEST_WAY, settings.TEST_SHOT, -1
        # ]

        # predictions = tf.math.log(tf.nn.softmax(distances, axis=1)), [
        #     settings.TRAIN_TEST_WAY, settings.TEST_SHOT, -1
        # ]

        # 修改的predictions
        predictions = tf.nn.softmax(-distances, axis=1)

        # 原论文的predictions
        # predictions = distances + tf.math.log(tf.nn.softmax(-distances, axis=1))
        # predictions /= (settings.TEST_SHOT * settings.TRAIN_TEST_WAY)

        return predictions

    def fn(self, y_true, y_pred):
        # y_pred = tf.reshape(y_pred, [settings.TRAIN_TEST_WAY, settings.TEST_SHOT, -1])
        # return tf.reduce_mean(tf.reshape(tf.squeeze(-tf.gather(y_pred, y_true, axis=2)), [-1]))
        return tf.reduce_mean()

    # @property
    # def loss_function(self):
    #     def wrapper(y_true, y_pred):
    #         if tf.is_tensor(y_pred) and tf.is_tensor(y_true):
    #             y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
    #                 y_pred, y_true
    #             )
    #
    #         ag_fn = tf.__internal__.autograph.tf_convert(
    #             self.fn, tf.__internal__.autograph.control_status_ctx()
    #         )
    #         return ag_fn(y_true, y_pred)
    #
    #     return wrapper

    @property
    def loss_function(self):
        return tf.keras.losses.MeanSquaredError()

    @property
    def trainable_variables(self):
        return self.encoder.trainable_variables

    # def labelEncode(self, labels):
    #     return labels

    def train(self, epochs, count_per_epoch):
        self.test(show=False)
        # self.encoder.load_weights("prototypical_show2.h5")
        super(PrototypicalNetwork, self).train(epochs, count_per_epoch)
        self.encoder.save_weights("maml_omniglot_5way_1shot.h5")
        self.test()
