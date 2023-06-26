import math

from dataLoader.miniDataLoader import MiniDataLoader
import matplotlib.pyplot as plt

dataLoader = MiniDataLoader()


def sampleDataset():
    return dataLoader.getDataset()


def showTest(images, labels, predictions, title=None, fig_text=None):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    images = images[:24]
    labels = labels[:24]
    predictions = predictions[:24]
    count = len(images)
    col = 6
    row = math.ceil(count / col)
    fig, axs = plt.subplots(nrows=row, ncols=col, figsize=(30, 30))
    fig.suptitle(title, fontsize=20, fontweight='bold')

    for i in range(count):
        plt.subplot(row, col, i + 1)
        plt.xticks([])
        plt.xlabel("T: {}, P: {}".format(labels[i], predictions[i]))
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
    plt.figtext(0.5, 0.01, fig_text, ha='center', fontsize=16)
    plt.show()
