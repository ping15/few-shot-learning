from configs import settings
import numpy as np
import scipy.io as sio
import tensorflow as tf
from sklearn.metrics import accuracy_score

BATCH_SIZE = 64

data_root = '../datas/cub/data'
dataset = 'CUB1_data'
image_embedding = 'res101'
class_embedding = 'original_att_splits'
mat_content = sio.loadmat(data_root + "/" + dataset + "/" + image_embedding + ".mat")
feature = mat_content['features'].T
label = mat_content['labels'].astype(int).squeeze() - 1
mat_content = sio.loadmat(data_root + "/" + dataset + "/" + class_embedding + ".mat")
# numpy array index starts from 0, matlab starts from 1
trainval_loc = mat_content['trainval_loc'].squeeze() - 1
test_seen_loc = mat_content['test_seen_loc'].squeeze() - 1
test_unseen_loc = mat_content['test_unseen_loc'].squeeze() - 1

attribute = mat_content['att'].T

x = feature[trainval_loc]  # train_features
train_label = label[trainval_loc].astype(int)  # train_label
att = attribute[train_label]  # train attributes

x_test = feature[test_unseen_loc]  # test_feature
test_label = label[test_unseen_loc].astype(int)  # test_label
x_test_seen = feature[test_seen_loc]  # test_seen_feature
test_label_seen = label[test_seen_loc].astype(int)  # test_seen_label
test_id = np.unique(test_label)  # test_id
att_pro = attribute[test_id]  # test_attribute

# train set
train_features = tf.convert_to_tensor(x, dtype=tf.float32)
# print(train_features.shape)

train_label = tf.convert_to_tensor(train_label)
# print(train_label.shape)

# attributes
all_attributes = np.array(attribute)
# print(all_attributes.shape)

attributes = tf.convert_to_tensor(attribute)
# test set

test_features = tf.convert_to_tensor(x_test, dtype=tf.float32)
# print(test_features.shape)

test_label = tf.expand_dims(tf.convert_to_tensor(test_label), axis=1)
# print(test_label.shape)

testclasses_id = np.array(test_id)
# print(testclasses_id.shape)

test_attributes = tf.convert_to_tensor(att_pro)
# print(test_attributes.shape)

test_seen_features = tf.convert_to_tensor(x_test_seen, dtype=tf.float32)
# print(test_seen_features.shape)

test_seen_label = tf.convert_to_tensor(test_label_seen)


class AttributeNetwork(tf.keras.models.Model):
    """docstring for RelationNetwork"""

    def __init__(self, hidden_size, output_size):
        super(AttributeNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation="relu")
        self.fc2 = tf.keras.layers.Dense(output_size, activation="relu")

    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return x


class RelationNetwork(tf.keras.models.Model):
    """docstring for RelationNetwork"""

    def __init__(self, hidden_size):
        super(RelationNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation="relu")
        self.fc2 = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return x


attribute_network = AttributeNetwork(1200, 2048)
relation_network = RelationNetwork(1200)

lossFn = tf.keras.losses.MeanSquaredError()
attribute_network_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, weight_decay=1e-5)
relation_network_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, weight_decay=1e-5)

def compute_accuracy(test_features, test_label, test_id, test_attributes):
    test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_label)).batch(32)

    total_rewards = 0
    # fetch attributes
    sample_labels = test_id
    sample_attributes = test_attributes
    class_num = sample_attributes.shape[0]
    test_size = test_features.shape[0]

    print("class num:", class_num)
    predict_labels_total = []
    re_batch_labels_total = []

    for batch_features, batch_labels in test_dataset:

        batch_size = batch_labels.shape[0]

        sample_features = attribute_network(sample_attributes)
        sample_features_ext = tf.repeat(tf.expand_dims(sample_features, axis=0), batch_size, axis=0)
        batch_features_ext = tf.repeat(tf.expand_dims(batch_features, axis=0), class_num, axis=0)
        batch_features_ext = tf.transpose(batch_features_ext, [1, 0, 2])

        # sample_features_ext = tf.repeat(tf.expand_dims(sample_features, axis=0), batch_size, axis=0)
        # batch_features_ext = tf.repeat(tf.expand_dims(batch_features, axis=1), class_num, axis=1)

        relation_pairs = tf.reshape(tf.concat([sample_features_ext, batch_features_ext], 2), [-1, 4096])
        relations = tf.reshape(relation_network(relation_pairs), [-1, class_num])

        # re-build batch_labels according to sample_labels

        re_batch_labels = []
        for label in batch_labels.numpy():
            index = np.argwhere(sample_labels == label)
            re_batch_labels.append(index[0][0])

        predict_labels = tf.argmax(relations, axis=1)
        predict_labels = np.array(predict_labels)
        re_batch_labels = np.array(re_batch_labels)

        predict_labels_total = np.append(predict_labels_total, predict_labels)
        re_batch_labels_total = np.append(re_batch_labels_total, re_batch_labels)

    # compute averaged per class accuracy
    predict_labels_total = np.array(predict_labels_total, dtype='int')
    re_batch_labels_total = np.array(re_batch_labels_total, dtype='int')
    unique_labels = np.unique(re_batch_labels_total)
    acc = 0
    for l in unique_labels:
        idx = np.nonzero(re_batch_labels_total == l)[0]
        acc += accuracy_score(re_batch_labels_total[idx], predict_labels_total[idx])
    acc = acc / unique_labels.shape[0]
    return acc


for epoch in range(200000):
    train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_label)) \
        .shuffle(len(train_features)).batch(BATCH_SIZE)
    for batch_features, batch_labels in train_dataset:
        with tf.GradientTape() as attribute_tape, tf.GradientTape() as relation_tape:
        # with tf.GradientTape() as tape:
            sample_labels = []
            for label in batch_labels.numpy():
                if label not in sample_labels:
                    sample_labels.append(label)

            sample_attributes = tf.convert_to_tensor([all_attributes[i] for i in sample_labels])
            class_num = sample_attributes.shape[0]

            sample_features = attribute_network(sample_attributes, training=True)  # k*312

            # print(sample_features.shape)
            # print(batch_features.shape)
            # print(tf.expand_dims(sample_features, axis=1).shape)
            sample_features_ext = tf.repeat(tf.expand_dims(sample_features, axis=0), BATCH_SIZE, axis=0)
            batch_features_ext = tf.repeat(tf.expand_dims(batch_features, axis=0), class_num, axis=0)
            batch_features_ext = tf.transpose(batch_features_ext, [1, 0, 2])

            # sample_features_ext = tf.repeat(tf.expand_dims(sample_features, axis=0), BATCH_SIZE, axis=0)
            # batch_features_ext = tf.repeat(tf.expand_dims(batch_features, axis=1), class_num, axis=1)

            relation_pairs = tf.reshape(tf.concat([sample_features_ext, batch_features_ext], axis=-1), [-1, 4096])
            # print(relation_pairs.shape)
            relations = tf.reshape(relation_network(relation_pairs, training=True), [-1, class_num])
            # print(relations.shape)

            sample_labels = np.array(sample_labels)
            re_batch_labels = []
            for label in batch_labels.numpy():
                index = np.argwhere(sample_labels == label)
                re_batch_labels.append(index[0][0])
            # predict_labels = tf.argmax(relations, axis=1)
            one_hot_labels = tf.one_hot(re_batch_labels, depth=class_num, axis=-1)

            loss = lossFn(one_hot_labels, relations)
            # loss = lossFn(re_batch_labels, predict_labels)
            print(one_hot_labels, relations)
            print(loss)
            # print(one_hot_labels, relations)

        attribute_gradients = attribute_tape.gradient(loss, attribute_network.trainable_variables)
        attribute_network_optimizer.apply_gradients(zip(attribute_gradients, attribute_network.trainable_variables))
        relation_gradients = relation_tape.gradient(loss, relation_network.trainable_variables)
        relation_network_optimizer.apply_gradients(zip(relation_gradients, relation_network.trainable_variables))

        # gradients = tape.gradient(loss, attribute_network.trainable_variables + relation_network.trainable_variables)
        # attribute_network_optimizer.apply_gradients(
        #     zip(gradients[:len(attribute_network.trainable_variables)], attribute_network.trainable_variables))
        # relation_network_optimizer.apply_gradients(
        #     zip(gradients[len(attribute_network.trainable_variables):], relation_network.trainable_variables))

        zsl_accuracy = compute_accuracy(test_features, test_label, test_id, test_attributes)
        gzsl_unseen_accuracy = compute_accuracy(test_features, test_label, np.arange(200), attributes)
        gzsl_seen_accuracy = compute_accuracy(test_seen_features, test_seen_label, np.arange(200), attributes)

        H = 2 * gzsl_seen_accuracy * gzsl_unseen_accuracy / (gzsl_unseen_accuracy + gzsl_seen_accuracy)

        print('zsl:', zsl_accuracy)
        print('gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_seen_accuracy, gzsl_unseen_accuracy, H))
        break

