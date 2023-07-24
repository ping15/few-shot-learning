import tensorflow as tf

# 定义学习率调度器
learning_rate_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001, decay_steps=100, alpha=0.001)

# 创建优化器，并设置学习率调度器
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_scheduler)

# 在训练循环中使用优化器
for epoch in range(100):
    print(learning_rate_scheduler(epoch))
