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


def print_metric(zsl, u, s, h):
    # 创建表格
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    table_data = [['指标', '值'], ['ZSL', zsl], ['U', u], ['S', s], ['H', h]]
    table = ax.table(cellText=table_data, loc='center')

    # 设置表格属性
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)

    # 显示表格
    plt.show()


print_metric(56.6, 38.91, 63.17, 48.15)
