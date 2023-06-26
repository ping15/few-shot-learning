import os
import random

import tensorflow as tf
import numpy as np
from PIL import Image

from .baseLoader import BaseDataLoader
from configs import settings

"""
class MiniDataLoader(BaseDataLoader):
    def __init__(self, basePath=settings.MINI_DATA_BASE_PATH):
        super(MiniDataLoader, self).__init__(basePath)

    def splitCSV(self, filename):
        dataFrame = pd.read_csv(os.path.join(self.basePath, filename))
        labels = np.array(dataFrame.pop("label"))
        imageNames = np.array(dataFrame)
        return imageNames, labels

    def getImagePath(self, imageNames):
        imagePaths = []
        for imageName in imageNames:
            imagePaths.append(os.path.join(self.basePath, "images", imageName[0]))

        return np.array(imagePaths)

    def labelMapping(self, imagePaths, labels):
        labelToIndex = {}
        indexToLabel = {}
        categoryDict = {}
        for imagePath, label in zip(imagePaths, labels):
            if label not in labelToIndex:
                labelToIndex[label] = len(labelToIndex)
                indexToLabel[len(indexToLabel)] = label

            if labelToIndex[label] not in categoryDict:
                categoryDict[labelToIndex[label]] = [imagePath]
            else:
                categoryDict[labelToIndex[label]].append(imagePath)

        return categoryDict, indexToLabel

    def processImages(self, imagePaths):
        pass

    def scale(self, images):
        pass

    def dataAugmentation(self, images):
        images = tf.image.random_brightness(images, 30)
        images = tf.image.random_contrast(images, lower=0.2, upper=1.8)
        images = tf.image.random_hue(images, max_delta=0.3)
        images = tf.image.random_saturation(images, lower=0.2, upper=1.8)
        return images

    def sample(self, labels, count):
        return random.sample(labels, count)

    def processImagePath(self, imagesPath):
        images = []
        for path in imagesPath:
            image = Image.open(path)
            image = image.convert('RGB')
            image = image.resize((42, 42))
            image = np.array(image).astype("float")
            image /= 255.0
            # image = tf.image.random_crop(image, (28, 28, 1))
            images.append(image)
        return images

    def sampleImages(self, labelsDict):
        # sampleLabels = random.sample(labels, settings.TRAIN_TEST_WAY)
        sampleLabels = random.choices(list(range(0, len(labelsDict))), k=settings.TRAIN_TEST_WAY)
        trainImages, trainLabels, testImages, testLabels = [], [], [], []
        for label in sampleLabels:
            imageList = labelsDict[label]
            imageList = random.sample(imageList, len(imageList))
            sampleTrainImagesPath, sampleTestImagesPath = imageList[:settings.TRAIN_SHOT], \
                                                          imageList[
                                                          settings.TRAIN_SHOT:settings.TRAIN_SHOT + settings.TEST_SHOT]
            sampleTrainImages = self.processImagePath(sampleTrainImagesPath)
            sampleTestImages = self.processImagePath(sampleTestImagesPath)

            for image in sampleTrainImages:
                trainImages.append(image)
                trainLabels.append(label)

            for image in sampleTestImages:
                testImages.append(image)
                testLabels.append(label)

        return np.array(trainImages), np.array(trainLabels), np.array(testImages), np.array(testLabels)

    def normalizeLabel(self, trainLabels, testLabels):
        rawLabels = list(collections.Counter(trainLabels).keys())
        newLabels = list(range(len(rawLabels)))

        labelMap = dict(zip(rawLabels, newLabels))

        for i in range(len(trainLabels)):
            trainLabels[i] = labelMap[trainLabels[i]]

        for i in range(len(testLabels)):
            testLabels[i] = labelMap[testLabels[i]]

        return trainLabels, testLabels


    def getDataset(self, filename="train.csv"):
        # 解析train_csv文件，将图片名字和标签进行分离
        imageNames, labels = self.splitCSV(filename="train.csv")

        # 将图片名字进行处理，得到图片绝对路径
        imagePaths = self.getImagePath(imageNames)

        # 将标签进行处理，将标签映射成数字，并生成标签到数字的字典
        labels, labelsMap = self.labelMapping(imagePaths, labels)

        # 从labels中抽取5个种类，从每个种类中抽取一张图片作为support set，抽取十张图片作为query set
        # 对support set和query set做处理生成images和labels
        trainImages, trainLabels, testImages, testLabels = self.sampleImages(labels)

        # 将图片进行数据增强
        # images = self.dataAugmentation(images)

        # 将query set进行打散
        trainImages, trainLabels, testImages, testLabels = self.shuffle(trainImages, trainLabels, testImages,
                                                                        testLabels)

        # trainImages = self.normalize(trainImages)
        # testImages = self.normalize(testImages)
        # trainImages = self.dataAugmentation(trainImages)
        # testImages = self.dataAugmentation(testImages)

        trainLabels, testLabels = self.normalizeLabel(trainLabels, testLabels)

        return trainImages, trainLabels, testImages, testLabels

    def normalize(self, images):
        return images / 255.0

    def shuffle(self, trainImages, trainLabels, testImages, testLabels):
        testIndexs = list(range(len(testImages)))
        random.shuffle(testIndexs)

        testImages = tf.gather(testImages, testIndexs)
        testLabels = tf.gather(testLabels, testIndexs)

        return trainImages, trainLabels, np.array(testImages), np.array(testLabels)

    def sampleDatasetFromTest(self):
        return self.getDataset(filename="test.csv")
"""


class MiniDataLoader(BaseDataLoader):
    def __init__(self, base_path=settings.MINI_DATA_BASE_PATH):
        super(MiniDataLoader, self).__init__(base_path)

    def total(self, folder_type):
        category_path_list = [os.path.join(self.basePath, folder_type, category) for category in \
                              os.listdir(os.path.join(self.basePath, folder_type))]
        count = 0
        for category_path in category_path_list:
            count += len(os.listdir(category_path))

        return count

    def get_image_folders(self, folder_type):
        category_path_list = [os.path.join(self.basePath, folder_type, category) for category in \
                              os.listdir(os.path.join(self.basePath, folder_type))]

        # characterPathList = []
        # for categoryPath in categoryPathList:
        #     characterPathList += [os.path.join(categoryPath, character) for character in os.listdir(categoryPath)]

        return category_path_list

    def split_character(self, all_characters):
        return all_characters[:settings.TRAIN_NUMBER], all_characters[settings.TRAIN_NUMBER:]

    def sample_characters(self, train_all_characters, ways_count):
        train_characters = random.sample(train_all_characters, ways_count)
        labels = list(range(ways_count))
        return train_characters, dict(zip(train_characters, labels))

    def get_class(self, sample):
        return os.path.join(*sample.split('\\')[:-1])

    def dataAugmentation(self, image):
        brightness = tf.random.uniform([], minval=0.9, maxval=1.1)
        contrast = tf.random.uniform([], minval=0.9, maxval=1.1)
        image = tf.image.adjust_brightness(image, delta=brightness)
        image = tf.image.adjust_contrast(image, contrast_factor=contrast)
        return image

    def normalize(self, image):
        return image / 255.0

    def get_images(self, folder_type="train"):
        image_path_folders = self.get_image_folders(folder_type=folder_type)[24: 32]

        images = []
        for image_path_folder in image_path_folders:
            image_path_list = [os.path.join(image_path_folder, imageName) for imageName in
                               os.listdir(image_path_folder)]

            for imagePath in image_path_list:
                image = Image.open(imagePath)
                image = np.array(image.convert("RGB")).astype("float")
                images.append(image)

        return np.stack(images)

    def sample_and_process_images(self, train_characters, labels_dict, start, shots_count, data_augmentation=False):
        images, labels = [], []
        for character_path in train_characters:
            image_path_list = [os.path.join(character_path, imageName) for imageName in os.listdir(character_path)]
            sample_images_path = image_path_list[start:shots_count]
            for path in sample_images_path:
                image = Image.open(path)
                image = image.convert('RGB')
                image = np.array(image)
                # normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
                # transform = transforms.Compose([transforms.ToTensor(), normalize])
                # image = transform(image).numpy()
                # image = np.transpose(image, (1, 2, 0))
                # image = np.asarray(image)
                image = image / 255.0

                # plt.figure()
                # plt.imshow(image)
                # plt.show()
                label = labels_dict[self.get_class(path)]
                image = image.astype("float")
                if data_augmentation:
                    image = self.dataAugmentation(image)
                # image = self.normalize(image)
                images.append(image)
                labels.append(label)

        return [np.array(images), np.array(labels)]

    def shuffle(self, train_images, train_labels, test_images, test_labels):
        test_indexes = list(range(len(test_images)))
        train_indexes = list(range(len(train_images)))
        random.shuffle(test_indexes)
        random.shuffle(train_indexes)

        train_images = tf.gather(train_images, train_indexes)
        train_labels = tf.gather(train_labels, train_indexes)

        test_images = tf.gather(test_images, test_indexes)
        test_labels = tf.gather(test_labels, test_indexes)

        return train_images, train_labels, test_images, test_labels

    def sample_dataset_from_test(self):
        return self.get_dataset(folder_type="test", data_augmentation=False)

    def shuffle_characters(self, characters):
        return random.sample(characters, len(characters))

    def sample_batch_dataset(self, batch_size, training=True):
        batch_train_images, batch_train_labels, batch_test_images, batch_test_labels = [], [], [], []

        for _ in range(batch_size):
            if training:
                train_images, train_labels, test_images, test_labels = self.get_dataset()
            else:
                train_images, train_labels, test_images, test_labels = self.sample_dataset_from_test()

            batch_train_images.append(train_images)
            batch_train_labels.append(train_labels)
            batch_test_images.append(test_images)
            batch_test_labels.append(test_labels)

        return np.stack(batch_train_images), np.stack(batch_train_labels), \
               np.stack(batch_test_images), np.stack(batch_test_labels)

    def get_dataset(self, folder_type="train", data_augmentation=False):
        # 获取所有的字符目录
        image_paths = self.get_image_folders(folder_type=folder_type)

        # 将字符目录切分为训练集和测试集
        train_all_characters, test_all_characters = self.split_character(image_paths)

        train_all_characters = self.shuffle_characters(train_all_characters)

        # 从训练集中随机抽取五个字符目录
        train_characters, labels_dict = self.sample_characters(train_all_characters, settings.TRAIN_TEST_WAY)

        # 每个字符目录随机抽取一张图片作为support set
        # 对support set做处理生成images和labels
        train_images, train_labels = self.sample_and_process_images(
            train_characters, labels_dict, 0, settings.TRAIN_SHOT)

        # 每个字符目录随机抽取十张图片作为query set
        # 对query set做处理生成images和labels
        test_images, test_labels = self.sample_and_process_images(
            train_characters, labels_dict, settings.TRAIN_SHOT,
            settings.TEST_SHOT + settings.TRAIN_SHOT)

        train_images, train_labels, test_images, test_labels = self.shuffle(
            train_images, train_labels, test_images, test_labels)

        return train_images, train_labels, test_images, test_labels

        # ImagePaths = self.shuffleCharacters(ImagePaths)
        #
        # # 从训练集中随机抽取五个字符目录
        # sampleImagePaths, labelsDict = self.sampleCharacters(ImagePaths, settings.TRAIN_TEST_WAY)
        #
        # # 每个字符目录随机抽取一张图片作为support set
        # # 对support set做处理生成images和labels
        # trainImages, trainLabels = self.sampleAndProcessImages(
        #     sampleImagePaths, labelsDict, 0, settings.TRAIN_SHOT,
        #     dataAugmentation=dataAugmentation
        # )
        #
        # # 每个字符目录随机抽取十张图片作为query set
        # # 对query set做处理生成images和labels
        # testImages, testLabels = self.sampleAndProcessImages(
        #     sampleImagePaths, labelsDict, settings.TRAIN_SHOT,
        #     settings.TEST_SHOT + settings.TRAIN_SHOT,
        #     dataAugmentation=dataAugmentation
        # )
        #
        # trainImages, trainLabels, testImages, testLabels = self.shuffle(
        #     trainImages, trainLabels, testImages, testLabels)
        #
        # return trainImages, trainLabels, testImages, testLabels

# meta_support_image, meta_support_label, meta_query_image, meta_query_label = next(train_data)
#         for support_image, support_label in zip(meta_support_image, meta_support_label):
#
#             # 每个task都需要载入最原始的weights进行更新
#             self.meta_model.set_weights(meta_weights)
#             for _ in range(inner_step):
#                 with tf.GradientTape() as tape:
#                     logits = self.meta_model(support_image, training=True)
#                     loss = losses.sparse_categorical_crossentropy(support_label, logits)
#                     loss = tf.reduce_mean(loss)
#
#                     acc = tf.cast(tf.argmax(logits, axis=-1, output_type=tf.int32) == support_label, tf.float32)
#                     acc = tf.reduce_mean(acc)
#
#                 grads = tape.gradient(loss, self.meta_model.trainable_variables)
#                 inner_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))
#
#             # 每次经过inner loop更新过后的weights都需要保存一次，保证这个weights后面outer loop训练的是同一个task
#             task_weights.append(self.meta_model.get_weights())
#
#         with tf.GradientTape() as tape:
#             for i, (query_image, query_label) in enumerate(zip(meta_query_image, meta_query_label)):
#
#                 # 载入每个task weights进行前向传播
#                 self.meta_model.set_weights(task_weights[i])
#
#                 logits = self.meta_model(query_image, training=True)
#                 loss = losses.sparse_categorical_crossentropy(query_label, logits)
#                 loss = tf.reduce_mean(loss)
#                 batch_loss.append(loss)
#
#                 acc = tf.cast(tf.argmax(logits, axis=-1) == query_label, tf.float32)
#                 acc = tf.reduce_mean(acc)
#                 batch_acc.append(acc)
#
#             mean_acc = tf.reduce_mean(batch_acc)
#             mean_loss = tf.reduce_mean(batch_loss)
#
#         # 无论是否更新，都需要载入最开始的权重进行更新，防止val阶段改变了原本的权重
#         self.meta_model.set_weights(meta_weights)
#         if outer_optimizer:
#             grads = tape.gradient(mean_loss, self.meta_model.trainable_variables)
#             outer_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))
#
#         return mean_loss, mean_acc
