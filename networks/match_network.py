import tensorflow as tf

fw_lstm_cells_encoder = [tf.keras.layers.LSTM(units=self.layer_sizes[i], activation=tf.nn.tanh)
                         for i in range(len(self.layer_sizes))]
bw_lstm_cells_encoder = [tf.keras.layers.LSTM(units=self.layer_sizes[i], activation=tf.nn.tanh)
                         for i in range(len(self.layer_sizes))]

bidirectional_layer = tf.keras.layers.Bidirectional(
    layer=tf.keras.layers.RNN(fw_lstm_cells_encoder + bw_lstm_cells_encoder, return_sequences=True),
    merge_mode="concat"
)

outputs = bidirectional_layer(inputs)
output_state_fw, output_state_bw = bidirectional_layer.state_size

import tensorflow as tf


class GEmbeddingBidirectionalLSTM(tf.keras.Model):
    def __init__(self, name, layer_sizes, batch_size):
        super(GEmbeddingBidirectionalLSTM, self).__init__(name=name)
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size

        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.RNN([tf.keras.layers.LSTMCell(units=self.layer_sizes[i], activation=tf.nn.tanh)
                                 for i in range(len(self.layer_sizes))]),
            merge_mode='concat'
        )

    def call(self, inputs, training=False):
        outputs = self.encoder(inputs)
        print("g out shape", outputs.shape)

        return outputs


# Usage:
batch_size = 32
layer_sizes = [64, 128, 256]
model = GEmbeddingBidirectionalLSTM(name="my_model", layer_sizes=layer_sizes, batch_size=batch_size)
inputs = tf.zeros(shape=(batch_size, 10, 32))
outputs = model(inputs)


class FEmbeddingBidirectionalLSTM(tf.keras.Model):
    def __init__(self, name, layer_size, batch_size):
        super(FEmbeddingBidirectionalLSTM, self).__init__(name=name)
        self.layer_size = layer_size
        self.batch_size = batch_size

        self.lstm_cell = tf.keras.layers.LSTMCell(units=self.layer_size, activation=tf.nn.tanh)
        self.attention_dense = tf.keras.layers.Dense(units=1, activation=tf.nn.softmax)

    def call(self, support_set_embeddings, target_set_embeddings, K, training=False):
        b, k, h_g_dim = support_set_embeddings.shape
        b, h_f_dim = target_set_embeddings.shape

        h = tf.zeros(shape=(b, h_g_dim))
        c_h = (h, h)
        c_h = (c_h[0], c_h[1] + target_set_embeddings)

        for i in range(K):
            attentional_softmax = tf.expand_dims(self.attention_dense(h), axis=2)
            attented_features = support_set_embeddings * attentional_softmax
            attented_features_summed = tf.reduce_sum(attented_features, axis=1)
            c_h = (c_h[0], c_h[1] + attented_features_summed)
            x, h_c = self.lstm_cell(inputs=target_set_embeddings, states=c_h)
            attentional_softmax = self.attention_dense(x)

        outputs = x
        print("out shape", outputs.shape)

        return outputs


# Usage:
batch_size = 32
layer_size = 64
K = 5
model = FEmbeddingBidirectionalLSTM(name="my_model", layer_size=layer_size, batch_size=batch_size)
support_set_embeddings = tf.zeros(shape=(batch_size, 10, 32))
target_set_embeddings = tf.zeros(shape=(batch_size, 32))
outputs = model(support_set_embeddings, target_set_embeddings, K)
