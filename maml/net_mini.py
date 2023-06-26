import random

import tensorflow as tf
import numpy as np

from dataLoader.miniDataLoader import MiniDataLoader
from dataLoader.utils import showTest


def block_res(x, output_channel):
    '''the forard function of BasicBlock give parametes'''

    residual = x

    out = tf.keras.layers.Conv2D(output_channel, 3, strides=1, padding="same")(x)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)

    out = tf.keras.layers.Conv2D(output_channel, 3, strides=1, padding="same")(out)
    out = tf.keras.layers.BatchNormalization()(out)

    # if downsample is True:
    #     residual = F.conv2d(x, params[base + 'downsample.0.weight'], stride=(1, 1))
    #     residual = F.batch_norm(residual, weight=params[base + 'downsample.1.weight'],
    #                             bias=params[base + 'downsample.1.bias'],
    #                             running_mean=modules['downsample']._modules['1'].running_mean,
    #                             running_var=modules['downsample']._modules['1'].running_var, training=mode)
    out += residual
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.MaxPool2D(2)(out)

    # if self.drop_rate > 0:
    #     if self.drop_block == True:
    #         feat_size = out.size()[2]
    #         keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
    #         gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
    #         out = self.DropBlock(out, gamma=gamma)
    #     else:
    #         out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

    return out


def block_conv(x, output_channel, kernel_size):
    x = tf.keras.layers.Conv2D(output_channel, kernel_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(output_channel, kernel_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    return x


def dropout(x, dropout_rate):
    return tf.keras.layers.Dropout(dropout_rate)(x)


class MAML:
    def __init__(self, input_shape, num_classes):
        """
        MAML模型类，需要两个模型，一个是作为真实更新的权重θ，另一个是用来做θ'的更新
        :param input_shape: 模型输入shape
        :param num_classes: 分类数目
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        # self.base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(84, 84, 3))
        # self.base_model.trainable = False
        self.meta_model = self.get_maml_model()
        self.data_loader = MiniDataLoader()

    def get_batch_data(self):
        return self.data_loader.sampleBatchDataset(8)

    def get_maml_model(self,):
        """
        建立maml模型
        :return: maml model
        """
        # inputs = tf.keras.layers.Input(shape=(84, 84, 3))
        #
        # x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
        #                            kernel_initializer=tf.keras.initializers.glorot_uniform())(inputs)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        #
        # x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
        #                            kernel_initializer=tf.keras.initializers.glorot_uniform())(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        #
        # x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
        #                            kernel_initializer=tf.keras.initializers.glorot_uniform())(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        #
        # x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
        #                            kernel_initializer=tf.keras.initializers.glorot_uniform())(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        #
        # x = layers.Flatten()(x)
        #
        # output = layers.Dense(self.num_classes, activation='softmax')(x)
        #
        # model = tf.keras.Model(inputs=inputs, outputs=output)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=1,
                                   kernel_initializer="he_normal",
                                   input_shape=self.input_shape),
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=1,
                                   kernel_initializer="he_normal"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=1,
                                   kernel_initializer="he_normal"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=1,
                                   kernel_initializer="he_normal"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.num_classes, activation='softmax',
                                  kernel_initializer="ones",
                                  bias_initializer="zeros"),
        ])

        return model

    def shuffle(self, images, labels):
        indexs = list(range(len(images)))
        random.shuffle(indexs)
        shuffle_images = tf.gather(images, indexs)
        shuffle_labels = tf.gather(labels, indexs)
        return shuffle_images, shuffle_labels

    def train_on_batch(self, train_data, inner_optimizer, inner_step,
                       outer_optimizer=None, train_step=True, painting=False):
        """
        MAML一个batch的训练过程
        :param painting: 000
        :param train_step: 是否在maml_test阶段
        :param train_data: 训练数据，以task为一个单位
        :param inner_optimizer: support set对应的优化器
        :param inner_step: 内部更新几个step
        :param outer_optimizer: query set对应的优化器，如果对象不存在则不更新梯度
        :return: batch query loss
        """
        batch_acc = []
        batch_loss = []
        task_weights = []

        # 用meta_weights保存一开始的权重，并将其设置为inner step模型的权重
        meta_weights = self.meta_model.get_weights()

        # meta_support_image, meta_support_label, meta_query_image, meta_query_label = self.get_batch_data()
        meta_support_image, meta_support_label, meta_query_image, meta_query_label = train_data
        # print(np.max(meta_query_image), np.min(meta_query_image))
        # print(np.max(meta_support_image), np.min(meta_support_image))
        # plt.figure(figsize=(4, 4))
        # for i in range(4):
        #     for j in range(4):
        #         plt.subplot(4, 4, i * 4 + j + 1)
        #         plt.imshow(meta_query_image[i][j])
        #         plt.xlabel(meta_query_label[i][j])
        # plt.show()
        for support_image, support_label in zip(meta_support_image, meta_support_label):
            # 每个task都需要载入最原始的weights进行更新
            self.meta_model.set_weights(meta_weights)
            for _ in range(inner_step):
                if not train_step:
                    support_image, support_label = self.shuffle(support_image, support_label)
                with tf.GradientTape() as tape:
                    logits = self.meta_model(support_image, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(support_label, logits)
                    loss = tf.reduce_mean(loss)

                    acc = tf.cast(tf.argmax(logits, axis=-1, output_type=tf.int32) == support_label, tf.float32)
                    acc = tf.reduce_mean(acc)

                grads = tape.gradient(loss, self.meta_model.trainable_variables)
                inner_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

            # 每次经过inner loop更新过后的weights都需要保存一次，保证这个weights后面outer loop训练的是同一个task
            task_weights.append(self.meta_model.get_weights())

        painting_list = [[] for _ in range(3)]
        with tf.GradientTape() as tape:
            for i, (query_image, query_label) in enumerate(zip(meta_query_image, meta_query_label)):
                # 载入每个task weights进行前向传播
                self.meta_model.set_weights(task_weights[i])
                # print(query_image.shape)
                logits = self.meta_model(query_image, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(query_label, logits)
                if painting:
                    painting_list[0].append(query_image)
                    painting_list[1].append(query_label)
                    painting_list[2].append(tf.argmax(logits, axis=-1))

                loss = tf.reduce_mean(loss)
                batch_loss.append(loss)

                acc = tf.cast(tf.argmax(logits, axis=-1) == query_label, tf.float32)
                acc = tf.reduce_mean(acc)
                batch_acc.append(acc)

            mean_acc = tf.reduce_mean(batch_acc)
            mean_loss = tf.reduce_mean(batch_loss)

        if painting:
            painting_list[0] = np.concatenate(painting_list[0], axis=0)
            painting_list[1] = np.concatenate(painting_list[1], axis=0)
            painting_list[2] = np.concatenate(painting_list[2], axis=0)
            showTest(painting_list[0], painting_list[1], painting_list[2])

        # 无论是否更新，都需要载入最开始的权重进行更新，防止val阶段改变了原本的权重
        self.meta_model.set_weights(meta_weights)
        if outer_optimizer:
            grads = tape.gradient(mean_loss, self.meta_model.trainable_variables)
            outer_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

        # if painting:
        #     self.meta_model.load_weights("../miniDatas/maml_mini.h5")
        #     _, _, meta_query_image, meta_query_label = self.get_batch_data()
        #     meta_query_image = meta_query_image[0]
        #     meta_query_label = meta_query_label[0]
        #     predictions = self.meta_model(meta_query_image)
        #     predictions = tf.argmax(predictions, axis=-1)
        #     showTest(meta_query_image, meta_query_label, predictions)

        return mean_loss, mean_acc
