import tensorflow as tf

a = tf.complex(tf.zeros(64), tf.ones(64))
b = tf.complex(tf.zeros(64), tf.ones(64))

print(a.shape, b.shape)
cos_distance = tf.keras.losses.CosineSimilarity(axis=0)
result = cos_distance(a, b)
print(result)
