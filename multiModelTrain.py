import os.path

import tensorflow as tf
import matplotlib.pyplot as plt

from configs import settings
from dataLoader import omniglotDataLoader
from models import RelationModel, CNNEncoder


encoderList = []
relationList = []

for _ in range(settings.MODEL_COUNT):
    encoder = CNNEncoder()
    relationNetwork = RelationModel(settings.RELATION_DIM)
    encoderList.append(encoder)
    relationList.append(relationNetwork)

dataLoader = omniglotDataLoader()

lossFn = tf.keras.losses.MeanSquaredError()
# optimizerList = [tf.keras.optimizers.Adam()] * settings.MODEL_COUNT
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")


def sampleDataset():
    return dataLoader.getDataset()

def showTest(images, labels, predictions):
    plt.figure(figsize=(20, 20))
    for i in range(50):
        plt.subplot(5, 10, i + 1)
        plt.xticks([])
        plt.xlabel("T: {}, P: {}".format(labels[i], predictions[i]))
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
    plt.show()

def forward(train_images, test_images, modelIndex):
    train_futures = encoderList[modelIndex](train_images, training=True)
    test_futures = encoderList[modelIndex](test_images, training=True)
    train_futures = tf.repeat(tf.expand_dims(train_futures, axis=0), settings.TEST_SHOT * settings.TRAIN_TEST_WAY,
                              axis=0)
    test_futures = tf.repeat(tf.expand_dims(test_futures, axis=1), settings.TRAIN_SHOT * settings.TRAIN_TEST_WAY,
                             axis=1)
    concat_futures = tf.concat([train_futures, test_futures], 4)
    concat_futures = tf.reshape(concat_futures, [-1, 5, 5, 128])

    relations = relationList[modelIndex](concat_futures, training=True)
    relations = tf.reshape(relations, [settings.TEST_SHOT * settings.TRAIN_TEST_WAY, settings.TRAIN_TEST_WAY])

    return relations

def labelEncode(labels):
    return tf.one_hot(labels, depth=settings.TRAIN_TEST_WAY, axis=1)

def test():
    test_accuracy.reset_state()
    for _ in range(128):
        # 采样一批数据集
        train_images, train_labels, test_images, test_labels = dataLoader.sampleDatasetFromTest()
        # predictions = tf.argmax(forward(train_images, test_images), axis=-1)

        # 遍历模型进行预测
        predictionsList = []
        for modelIndex in range(settings.MODEL_COUNT):
            predictionsList.append(tf.argmax(forward(train_images, test_images, modelIndex), axis=-1))

        predictionsConcat = [[] for _ in range(len(predictionsList[0]))]
        for predictions in predictionsList:
            for i, prediction in enumerate(predictions):
                predictionsConcat[i].append(prediction)

        predictions = []
        for multiPrediction in predictionsConcat:
            predictions.append(max(multiPrediction, key=multiPrediction.count))

        oneHotPredictions = labelEncode(predictions)
        oneHotLabels = labelEncode(test_labels)

        test_accuracy(oneHotLabels, oneHotPredictions)
    print("test_accuracy={:.4f}".format(test_accuracy.result()))


    # showTest(test_images, test_labels, predictions)

test()





def train_step(train_images, train_labels, test_images, test_labels, modelIndex):
    with tf.GradientTape() as tape:
        relations = forward(train_images, test_images, modelIndex)
        oneHotLabels = labelEncode(test_labels)
        loss = lossFn(oneHotLabels, relations)
    gradiens = tape.gradient(loss, encoderList[modelIndex].trainable_variables + relationList[modelIndex].trainable_variables)
    optimizer.apply_gradients(zip(gradiens, encoderList[modelIndex].trainable_variables + relationList[modelIndex].trainable_variables))

    train_loss(loss)
    train_accuracy(oneHotLabels, relations)

def test_step(train_images, train_labels, test_images, test_labels, modelIndex):
    relations = forward(train_images, test_images, modelIndex)
    oneHotLabels = labelEncode(test_labels)
    loss = lossFn(oneHotLabels, relations)

    test_loss(loss)
    test_accuracy(oneHotLabels, relations)

COUNT = 128
EPOCHS = 40
WEIGHTS_SAVE_PATH = "./weights"
ENCODER_WEIGHTS_SAVE_NAME = "omniglot_encoder_{:03d}.ckpt"
RELATION_WEIGHTS_SAVE_NAME = "omniglot_relation_{:03d}.ckpt"
for modelIndex in range(settings.MODEL_COUNT):
    optimizer = tf.keras.optimizers.Adam()
    # optimizer.build(encoderList[modelIndex].trainable_variables + relationList[modelIndex].trainable_variables)
    for epoch in range(EPOCHS):
        train_loss.reset_state()
        train_accuracy.reset_state()
        test_loss.reset_state()
        test_accuracy.reset_state()

        for _ in range(COUNT):
            train_images, train_labels, test_images, test_labels = sampleDataset()
            train_step(train_images, train_labels, test_images, test_labels, modelIndex)

        for _ in range(COUNT // 2):
            train_images, train_labels, test_images, test_labels = sampleDataset()
            test_step(train_images, train_labels, test_images, test_labels, modelIndex)

        print(
            "Epoch: {:.2f} ".format(epoch + 1),
            "ModelIndex: {} ".format(modelIndex),
            "train_loss: {:.2f} ".format(train_loss.result()),
            "train_accuracy: {:.4f} ".format(train_accuracy.result()),
            "test_loss: {:.2f} ".format(test_loss.result()),
            "test_accuracy: {:.4f} ".format(test_accuracy.result())
        )

    encoderList[modelIndex].save_weights(os.path.join(WEIGHTS_SAVE_PATH, ENCODER_WEIGHTS_SAVE_NAME.format(modelIndex)))
    relationList[modelIndex].save_weights(os.path.join(WEIGHTS_SAVE_PATH, RELATION_WEIGHTS_SAVE_NAME.format(modelIndex)))

test()