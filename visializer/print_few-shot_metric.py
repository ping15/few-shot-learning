import pickle

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + 1e-13))


def print_metric(y_true, y_pred):
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred)

    # 创建指标类
    accuracy = tf.keras.metrics.Accuracy()
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    auc = tf.keras.metrics.AUC()
    f1_score = F1Score()

    # 更新指标值
    f1_score.update_state(y_true, y_pred)
    accuracy.update_state(y_true, y_pred)
    precision.update_state(y_true, y_pred)
    recall.update_state(y_true, y_pred)
    f1_score.update_state(y_true, y_pred)
    auc.update_state(y_true, y_pred)

    # 创建数据
    accuracy = accuracy.result().numpy()
    precision = precision.result().numpy()
    recall = recall.result().numpy()
    f1_score = f1_score.result().numpy()
    auc = auc.result().numpy()

    # 创建表格
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    table_data = [['指标', '值'], ['准确率', accuracy], ['精确率', precision], ['召回率', recall],
                  ['F1分数', f1_score], ['AUC', auc]]
    table = ax.table(cellText=table_data, loc='center')

    # 设置表格属性
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)

    # plt.title("关系网络在测试集的指标")
    # plt.figtext(0.5, 0.01, "Figure(2-9)", ha='center', fontsize=16)

    # 显示表格
    plt.show()


with open("maml_omniglot_5way_1shot_label_prediction.pkl", "rb") as f:
    labels_list, predictions_list = pickle.load(f)
    labels_list = labels_list[-10]
    labels_list = tf.argmax(np.array(labels_list), axis=-1)
    predictions_list = predictions_list[-10]
    predictions_list = tf.argmax(np.array(predictions_list), axis=-1)

print_metric(labels_list, predictions_list)
