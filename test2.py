import tensorflow as tf


class EmbeddingAttentionModel(tf.keras.Model):
    def __init__(self, input_dim):
        super(EmbeddingAttentionModel, self).__init__()
        self.embedding = tf.keras.layers.Dense(input_dim)
        self.attention = tf.keras.layers.Attention()
        self.dense = tf.keras.layers.Dense(input_dim, activation='relu')

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)

        attn_output = self.attention([x, x])

        concat_output = tf.concat([x, attn_output], axis=-1)

        output = self.dense(concat_output)

        return output


input_dim = 64  # 输入的维度

# 创建模型实例
model = EmbeddingAttentionModel(input_dim)

# 输入数据
batch_size = 32
seq_length = 15
inputs = tf.random.uniform((batch_size, seq_length, input_dim), dtype=tf.float32)

# 前向传播
outputs = model(inputs)

print("输入的shape:", inputs.shape)
print("输出的shape:", outputs.shape)
