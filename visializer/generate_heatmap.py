import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei',font_scale=1.5)  # 解决Seaborn中文显示问题并调整字体大小

with open("prototypical_omniglot_5way_1shot_label_prediction.pkl", "rb") as f:
    labels_list, predictions_list = pickle.load(f)
    labels_list = labels_list[-1][-15:-10]
    predictions_list = predictions_list[-1][-15:-10]

# 创建混淆矩阵
# plt.subplot(2, 1, 1)
label_conf_matrix = np.array(labels_list)

# 绘制热度图
sns.set(font_scale=1.4)
sns.heatmap(label_conf_matrix, annot=True, annot_kws={"size": 16}, cmap="Blues",
            xticklabels=['', '', '', '', ''],
            yticklabels=['', '', '', '', ''])
plt.tick_params(axis='both', which='both', length=0)
# plt.ylabel('index')
# plt.xlabel('class')
# plt.title("Relation Network Labels")
# plt.figtext(0.5, 0.01, "Figure(2-7)", ha='center', fontsize=16)
plt.show()

# plt.subplot(2, 1, 2)
pred_conf_matrix = np.array(predictions_list)

# 绘制热度图
sns.set(font_scale=1.4)
sns.heatmap(pred_conf_matrix, annot=True, annot_kws={"size": 16}, cmap="Blues",
            xticklabels=['', '', '', '', ''],
            yticklabels=['', '', '', '', ''])
plt.tick_params(axis='both', which='both', length=0)

plt.show()
