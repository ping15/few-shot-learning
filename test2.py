import tensorflow as tf

inputs = tf.random.normal((32, 16))
states = tf.random.normal((32, 64))
lstm = tf.keras.layers.LSTMCell(64)
outputs = lstm(inputs, states=states)
print(outputs.shape)
