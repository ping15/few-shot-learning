import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 加载数据集
(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()

# 对数据进行预处理
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255.0
test_images /= 255.0

# 将标签转换为 one-hot 编码
train_labels = tf.keras.utils.to_categorical(train_labels, 100)
test_labels = tf.keras.utils.to_categorical(test_labels, 100)

# 构建模型
model = Sequential()

# 第一层卷积层，使用64个3x3的卷积核，padding为same，激活函数为relu
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
# 第二层卷积层，使用64个3x3的卷积核，padding为same，激活函数为relu
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# 第一层池化层，使用2x2的池化核，步长为2
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# 第一层批量归一化层
model.add(BatchNormalization())

# 第三层卷积层，使用128个3x3的卷积核，padding为same，激活函数为relu
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
# 第四层卷积层，使用128个3x3的卷积核，padding为same，激活函数为relu
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
# 第二层池化层，使用2x2的池化核，步长为2
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# 第二层批量归一化层
model.add(BatchNormalization())

# 第五层卷积层，使用256个3x3的卷积核，padding为same，激活函数为relu
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
# 第六层卷积层，使用256个3x3的卷积核，padding为same，激活函数为relu
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
# 第七层卷积层，使用256个3x3的卷积核，padding为same，激活函数为relu
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
# 第三层池化层，使用2x2的池化核，步长为2
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# 第三层批量归一化层
model.add(BatchNormalization())

# 第八层卷积层，使用512个3x3的卷积核，padding为same，激活函数为relu
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# 第九层卷积层，使用512个3x3的卷积核，padding为same，激活函数为relu
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# 第十层卷积层，使用512个3x3的卷积核，padding为same，激活函数为relu
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# 第四层池化层，使用2x2的池化核，步长为2
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# 第四层批量归一化层
model.add(BatchNormalization())

# 第十一层卷积层，使用512个3x3的卷积核，padding为same，激活函数为relu
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# 第十二层卷积层，使用512个3x3的卷积核，padding为same，激活函数为relu
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# 第十三层卷积层，使用512个3x3的卷积核，padding为same，激活函数为relu
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# 第五层池化层，使用2x2的池化核，步长为2
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# 第五层批量归一化层
model.add(BatchNormalization())

# 将特征图转换为一维向量
model.add(Flatten())
# 第一层全连接层，使用4096个神经元，激活函数为relu，添加dropout
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
# 第二层全连接层，使用4096个神经元，激活函数为relu，添加dropout
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
# 第三层全连接层，使用100个神经元，激活函数为softmax
model.add(Dense(100, activation='softmax'))

# 输出模型的结构信息
model.summary()

# 编译模型，使用Adam优化器，学习率为1e-4，损失函数为交叉熵，评估指标为准确率
adam = Adam(lr=1e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# 对训练集进行数据增强
datagen = ImageDataGenerator(
    rotation_range=15,  # 随机旋转图像的角度范围
    width_shift_range=0.1,  # 随机水平平移图像的范围
    height_shift_range=0.1,  # 随机垂直平移图像的范围
    horizontal_flip=True,  # 随机将图像水平翻转
    zoom_range=0.1)  # 随机缩放图像的范围

# 训练模型，使用数据增强，每个epoch训练完后在验证集上进行验证
history = model.fit(datagen.flow(train_images, train_labels, batch_size=64),
                    epochs=100,
                    validation_data=(test_images, test_labels),
                    verbose=1)

# 在测试集上评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print('Test accuracy:', test_acc)