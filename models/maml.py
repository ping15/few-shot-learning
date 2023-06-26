import tensorflow as tf

from configs import settings

def conv_block(inputs, filters, kernel_size):
    x = tf.layers.conv2d(inputs, filters, kernel_size, padding='same')
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2), padding='same')
    return x


def block_conv(output_channel, kernel_size):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=output_channel, kernel_size=kernel_size, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(2),
    ])


def maml_model():
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = block_conv(64, 3)(inputs)
    x = block_conv(64, 3)(x)
    x = block_conv(64, 3)(x)
    x = block_conv(64, 3)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(settings.TRAIN_TEST_WAY)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model


class MAMLModel(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super(MAMLModel, self).__init__(*args, **kwargs)
        self.layer1 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="valid"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(),
        ])
        self.layer2 = block_conv(64, 3)
        self.layer3 = block_conv(64, 3)
        self.layer4 = block_conv(64, 2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(settings.TRAIN_TEST_WAY, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        return self.dense(x)
