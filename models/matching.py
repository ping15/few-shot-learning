import time
from functools import wraps

import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils


def timeit(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        fn(*args, **kwargs)
        print(f"花费了{time.time() - start}秒")

    return wrapper


class CNNEncoder(tf.keras.models.Model):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                64, kernel_size=3, padding="valid",
            ),
            tf.keras.layers.BatchNormalization(momentum=1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(2),
        ])
        self.layer2 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                64, kernel_size=3, padding="valid",
            ),
            tf.keras.layers.BatchNormalization(momentum=1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(2),
        ])
        self.layer3 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                64, kernel_size=3, padding="valid",
            ),
            tf.keras.layers.BatchNormalization(momentum=1),
            tf.keras.layers.ReLU(),
        ])
        self.layer4 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                64, kernel_size=3, padding="valid",
            ),
            tf.keras.layers.BatchNormalization(momentum=1),
            tf.keras.layers.ReLU(),
        ])
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: [batch_size, n, 28, 28, 1]
        :return: [batch_size, n, 64]
        """
        batch_size, n, image_height, image_weight, image_channel = inputs.shape
        x = tf.reshape(inputs, (-1, image_height, image_weight, image_channel))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = tf.reshape(x, (batch_size, n, -1))
        return x

    @classmethod
    @timeit
    def test(cls):
        """测试代码"""
        batch_images = tf.random.normal((32, 21, 28, 28, 1))
        res = cls()(batch_images)
        assert res.shape == (32, 21, 64)


class GEmbeddingBidirectionalLSTM(tf.keras.models.Model):
    def __init__(self, layer_sizes, batch_size):
        super(GEmbeddingBidirectionalLSTM, self).__init__()
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size

        self.encoder = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(
            [tf.keras.layers.LSTMCell(units=self.layer_sizes[i]) for i in range(len(self.layer_sizes))],
            return_sequences=True
        ))

    def call(self, inputs, training=None, mask=None, *args, **kwargs):
        """
        :param inputs: [batch_size, class_num * num_per_class, 64]
        :return: [batch_size, class_num * num_per_class, 64]
        """
        outputs = self.encoder(inputs)

        return outputs

    @classmethod
    @timeit
    def test(cls):
        """测试代码"""
        query_embeddings = tf.random.normal((32, 4 * 5, 64))
        res = cls([32, 32, 32], 32)(query_embeddings)
        assert res.shape == (32, 20, 64)


class FEmbeddingBidirectionalLSTM(tf.keras.models.Model):
    def __init__(self, units):
        super(FEmbeddingBidirectionalLSTM, self).__init__()
        self.lstm = tf.keras.layers.LSTM(units, return_sequences=True)

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: [batch_size, class_num * num_per_class, 64]
        :return: [batch_size, class_num * num_per_class, 64]
        """
        return self.lstm(inputs)

    @classmethod
    @timeit
    def test(cls):
        """测试代码"""
        query_embeddings = tf.random.normal((32, 4 * 5, 64))
        res = cls(64)(query_embeddings)
        assert res.shape == (32, 20, 64)


class DistanceNetwork:
    def __init__(self, num_class):
        self.num_class = num_class

    def __call__(self, support_embeddings, support_labels, query_embeddings, *args, **kwargs):
        """
        :param support_embeddings: [batch_size, class_num * num_per_class, 64]
        :param support_labels: [batch_size, class_num * num_per_class]
        :param query_embeddings: [batch_size, query_number, 64]
        :return: output: [batch_size, query_number, class_number]
        """

        # # 暂时没什么好的招（可优化）
        # cosine_similarity = tf.keras.losses.CosineSimilarity(axis=-1, reduction=losses_utils.ReductionV2.NONE)
        # all_batch_score_list = []
        # for batch_idx, (support_task, query_task) in enumerate(zip(support_embeddings, query_embeddings)):
        #     batch_score_list = []
        #     for query_idx, query_embedding in enumerate(query_task):
        #         score_arr = np.array([0 for _ in range(self.num_class)], dtype=np.float32)
        #         for support_idx, support_embedding in enumerate(support_task):
        #             score = -cosine_similarity(query_embedding, support_embedding)
        #             score_arr[int(support_labels[batch_idx][support_idx])] += score
        #         batch_score_list.append(score_arr)
        #     all_batch_score_list.append(np.stack(batch_score_list))
        # all_batch_score_arr = np.stack(all_batch_score_list)
        # return all_batch_score_arr

        # 优化后
        cosine_similarity = tf.keras.losses.CosineSimilarity(axis=-1, reduction=losses_utils.ReductionV2.NONE)
        # cosine_similarity = tf.keras.metrics.CosineSimilarity(axis=-1)
        all_batch_score_list = []
        for batch_idx, (support_task, query_task) in enumerate(zip(support_embeddings, query_embeddings)):
            batch_score_list = []
            for query_idx, query_embedding in enumerate(query_task):
                score_arr = tf.constant([0 for _ in range(self.num_class)], dtype=tf.float32)
                scores = -cosine_similarity(query_embedding, support_task)
                batch_score_list.append(tf.tensor_scatter_nd_add(
                    score_arr,
                    tf.expand_dims(tf.cast(support_labels[batch_idx], tf.int32), axis=1),
                    scores
                ))
            all_batch_score_list.append(tf.stack(batch_score_list))
        all_batch_score_arr = tf.stack(all_batch_score_list)
        return all_batch_score_arr

    @classmethod
    @timeit
    def test(cls):
        """测试代码"""
        support_embeddings = tf.random.normal((32, 3 * 5, 64))
        support_labels = tf.ones((32, 3 * 5))
        query_embeddings = tf.random.normal((32, 21, 64))

        res = cls(5)(support_embeddings, support_labels, query_embeddings)
        assert res.shape == (32, 21, 5)


if __name__ == "__main__":
    CNNEncoder.test()
    GEmbeddingBidirectionalLSTM.test()
    FEmbeddingBidirectionalLSTM.test()
    DistanceNetwork.test()
