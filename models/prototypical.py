import tensorflow as tf


def conv_block(output_channels):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(output_channels, 3, padding="same",
                               kernel_initializer="he_normal"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(2),
        # tf.keras.layers.Dropout(0.4),
    ])


def conv_block_no_down_sample(output_channels):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(output_channels, 3, padding="same",
                               kernel_initializer="he_normal"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.5),
    ])


class Encoder(tf.keras.models.Model):
    def __init__(self, hidden_dim=64, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        self.layer1 = conv_block(hidden_dim)

        self.layer2 = conv_block(hidden_dim)

        self.layer3 = conv_block(hidden_dim)

        self.layer4 = conv_block(hidden_dim)

        self.flatten = tf.keras.layers.Flatten()

        # self.dense1 = tf.keras.layers.Dense(128, activation="relu")

        # self.dense2 = tf.keras.layers.Dense(128, activation="relu")
        # self.flatten = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        # x = self.dense1(x)
        # x = self.dense2(x)
        return x
