import tensorflow as tf

from maml.net import MAML

maml = MAML(input_shape=(28, 28, 1), num_classes=5)

# 打印模型结构并保存为图片
tf.keras.utils.plot_model(maml.meta_model, to_file='model.png', show_shapes=True, show_layer_names=True)
