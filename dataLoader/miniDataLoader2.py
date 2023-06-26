import os
import random

from PIL import Image
import numpy as np
import tensorflow as tf

from configs import settings
from .baseLoader import BaseDataLoader
from configs import settings


class MiniImageDataLoader(BaseDataLoader):
    def __init__(self, base_path=settings.MINI_DATA_BASE_PATH):
        super(MiniImageDataLoader, self).__init__(base_path)

    def get_all_character_path(self, data_type="train"):
        category_path = os.path.join(self.basePath, data_type)

        character_path_list = [os.path.join(category_path, character) for character in os.listdir(category_path)]

        # print(f"len={len(characterPathList)}")
        return character_path_list

    def split_character(self, all_characters):
        return all_characters[:settings.TRAIN_NUMBER], all_characters[settings.TRAIN_NUMBER:]

    def sample_characters(self, train_all_characters, ways_count):
        train_characters = random.sample(train_all_characters, ways_count)
        labels = list(range(ways_count))
        return train_characters, dict(zip(train_characters, labels))

    def get_class(self, sample):
        return os.path.join(*sample.split('\\')[:-1])

    def data_augmentation(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        # image = tf.image.resize(image, (36, 36))
        # image = tf.image.random_crop(image, (28, 28, 1))
        return image

    def normalize(self, image):
        return image / 255.0

    def sample_and_process_images(self, train_characters, labels_dict, start, shots_count, data_augmentation=True):
        images, labels = [], []
        row_shots_count = shots_count
        for character_path in train_characters:
            shots_count = row_shots_count
            image_path_list = [os.path.join(character_path, imageName) for imageName in os.listdir(character_path)]
            image_path_list = self.shuffle_characters(image_path_list)
            sample_images_path = image_path_list[start: shots_count]
            while shots_count > len(image_path_list) + 1:
                shots_count -= (len(image_path_list) - start)
                sample_images_path += image_path_list[start: shots_count]
            for path in sample_images_path:
                image = Image.open(path)
                image = image.convert('RGB')
                label = labels_dict[self.get_class(path)]
                image = np.array(image).astype("float")
                # if dataAugmentation:
                #     image = self.dataAugmentation(image)
                image = self.normalize(image)
                images.append(image)
                labels.append(label)

        return [np.array(images), np.array(labels)]

    def shuffle(self, train_images, train_labels, test_images, test_labels):
        test_indexes = list(range(len(test_images)))
        random.shuffle(test_indexes)

        test_images = tf.gather(test_images, test_indexes)
        test_labels = tf.gather(test_labels, test_indexes)

        return train_images, train_labels, test_images, test_labels

    def sample_dataset_from_test(self):
        # 获取所有的字符目录
        all_characters = self.get_all_character_path()

        # 将字符目录切分为训练集和测试集
        train_all_characters, test_all_characters = self.split_character(all_characters)

        # 从训练集中随机抽取五个字符目录
        train_characters, labels_dict = self.sample_characters(test_all_characters, settings.TRAIN_TEST_WAY)

        # 每个字符目录随机抽取一张图片作为support set
        # 对support set做处理生成images和labels
        train_images, train_labels = self.sample_and_process_images(
            train_characters, labels_dict, 0, settings.TRAIN_SHOT, data_augmentation=False)

        # 每个字符目录随机抽取十张图片作为query set
        # 对query set做处理生成images和labels
        test_images, test_labels = self.sample_and_process_images(
            train_characters, labels_dict, settings.TRAIN_SHOT,
            settings.TEST_SHOT + settings.TRAIN_SHOT, data_augmentation=False)

        # 将query set进行打散
        train_images, train_labels, test_images, test_labels = self.shuffle(
            train_images, train_labels, test_images, test_labels)

        return train_images, train_labels, test_images, test_labels

    def shuffle_characters(self, characters):
        return random.sample(characters, len(characters))

    def sample_batch_dataset(self, batch_size, training=True, resize=False):
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

        batch_train_images, batch_train_labels, batch_test_images, batch_test_labels = \
            np.stack(batch_train_images), np.stack(batch_train_labels), \
            np.stack(batch_test_images), np.stack(batch_test_labels)
        if resize:
            batch_train_images_shape = batch_train_images.shape
            batch_test_images_shape = batch_test_images.shape
            batch_train_images = batch_train_images.reshape([
                -1,
                batch_train_images_shape[-3],
                batch_train_images_shape[-2],
                batch_train_images_shape[-1]])
            batch_train_labels = batch_train_labels.reshape([-1])
            batch_test_images = batch_test_images.reshape([
                -1,
                batch_test_images_shape[-3],
                batch_test_images_shape[-2],
                batch_test_images_shape[-1]])
            batch_test_labels = batch_test_labels.reshape([-1])
        return batch_train_images, batch_train_labels, batch_test_images, batch_test_labels

    def get_dataset(self):
        # 获取所有的字符目录
        all_characters = self.get_all_character_path()

        # 将字符目录切分为训练集和测试集
        train_all_characters, test_all_characters = self.split_character(all_characters)

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

    def get_images(self):
        image_path_folders = self.get_all_character_path()

        images = []
        for image_path_folder in image_path_folders:
            image_path_list = [os.path.join(image_path_folder, image_name) for image_name in os.listdir(image_path_folders)]

            for image_path in image_path_list:
                image = Image.open(image_path)
                image = image.convert('L')
                image = image.resize((28, 28), resample=Image.LANCZOS)
                image = np.array(image).astype("float")[..., tf.newaxis]
                # if dataAugmentation:
                #     image = self.dataAugmentation(image)
                image = self.normalize(image)
                images.append(image)
        return np.stack(images)
