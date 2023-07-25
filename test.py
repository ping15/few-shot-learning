import tensorflow as tf
import numpy as np


# 定义余弦退火元优化器学习率函数
def cosine_annealing_lr(epoch, max_epochs, initial_lr, min_lr):
    cos_inner = np.pi * (epoch % (max_epochs // 2))
    cos_inner /= max_epochs // 2
    cos_out = np.cos(cos_inner) + 1
    lr = (initial_lr - min_lr) / 2 * cos_out + min_lr
    return lr


# 设置训练参数
max_epochs = 100
initial_lr = 0.001
min_lr = 0.001

# 创建TensorFlow变量
epoch_var = tf.Variable(0, trainable=False, dtype=tf.int64)
lr_var = tf.Variable(initial_lr, trainable=False, dtype=tf.float32)

# 定义训练循环
for epoch in range(max_epochs):
    # 更新学习率
    lr = cosine_annealing_lr(epoch, max_epochs, initial_lr, min_lr)
    lr_var.assign(lr)
    print(lr_var)

    # 在这里执行模型训练的代码
    # ...

    # 更新训练周期
    epoch_var.assign(epoch + 1)
