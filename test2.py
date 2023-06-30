# import tensorflow as tf
#
# # 使用随机数据创建输入张量
# from tensorflow.python.keras.utils import losses_utils
#
# input1 = tf.random.normal((21, 32, 64))
# input2 = tf.random.normal((64, ))
#
# # 计算余弦相似性
# similarity = tf.keras.losses.CosineSimilarity(axis=-1, reduction=losses_utils.ReductionV2.NONE)
# result = similarity(input1, input2)
#
# print(result.shape)  # 输出: (32, )

# import tensorflow as tf
#
# # 定义输入数组
# a = tf.Variable([0, 0, 0, 0, 0], dtype=tf.float32)
# b = tf.constant([1, 3, 4, 2, 2, 4, 0], dtype=tf.int32)
# c = tf.constant([-1, 0.5, 0.2, 0.3, 0.4, 0.3, 0.3], dtype=tf.float32)
#
# # [0.3, -1, 0.7, 0.5, 0.5]
# # 根据索引 b 和对应值 c 更新数组 a
# d = tf.tensor_scatter_nd_add(a, tf.expand_dims(b, axis=1), c)
#
# # 打印更新后的数组 a
# print(d)

import tensorflow as tf

a = tf.constant([1, 2, 3])
print(a.shape)
