import tensorflow as tf


class CNNEncoder(tf.keras.models.Model):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                64, kernel_size=3, padding="valid",
                kernel_initializer=tf.keras.initializers.random_normal(0, 2. / (64 * 3 * 3)),
                # kernel_initializer="he_normal",
            ),
            tf.keras.layers.BatchNormalization(momentum=1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(2),
        ])
        self.layer2 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                64, kernel_size=3, padding="valid",
                kernel_initializer=tf.keras.initializers.random_normal(0, 2. / (64 * 3 * 3)),
                # kernel_initializer="he_normal",
            ),
            tf.keras.layers.BatchNormalization(momentum=1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(2),
        ])
        self.layer3 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                64, kernel_size=3, padding="same",
                kernel_initializer=tf.keras.initializers.random_normal(0, 2. / (64 * 3 * 3)),
                # kernel_initializer="he_normal",
            ),
            tf.keras.layers.BatchNormalization(momentum=1),
            tf.keras.layers.ReLU(),
        ])
        self.layer4 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                64, kernel_size=3, padding="same",
                kernel_initializer=tf.keras.initializers.random_normal(0, 2. / (64 * 3 * 3)),
                # kernel_initializer="he_normal",
            ),
            tf.keras.layers.BatchNormalization(momentum=1),
            tf.keras.layers.ReLU(),
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class RelationModel(tf.keras.models.Model):
    def __init__(self, hidden_size):
        super(RelationModel, self).__init__()
        self.layer1 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                64, kernel_size=3, padding="same",
                kernel_initializer=tf.keras.initializers.random_normal(0, 2. / (64 * 3 * 3)),
                # kernel_initializer="he_normal",
            ),
            tf.keras.layers.BatchNormalization(momentum=1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(2),
            # tf.keras.layers.Dropout(0.2),
        ])
        self.layer2 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                64, kernel_size=3, padding="same",
                kernel_initializer=tf.keras.initializers.random_normal(0, 2. / (64 * 3 * 3)),
                # kernel_initializer="he_normal",
            ),
            tf.keras.layers.BatchNormalization(momentum=1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(2),
            # tf.keras.layers.Dropout(0.2),
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.random_normal(0, 0.01),
            activation="relu",
            bias_initializer="ones",
        )
        self.dense2 = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
