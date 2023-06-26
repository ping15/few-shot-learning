from keras import optimizers, utils
import numpy as np

from dataLoader.miniDataLoader import MiniDataLoader
from net_mini import MAML
from config_mini import args
import shutil
import os

import random
import sys
import time
import threading

from PySide6.QtCore import Qt, Slot, QObject, Signal
from PySide6.QtCharts import QSplineSeries, QValueAxis, QChart, QChartView
from PySide6.QtCore import QTimer
from PySide6.QtGui import QPainter, QFont, QColor
from PySide6.QtWidgets import QMainWindow, QApplication
import tensorflow as tf


class DynamicChart(QMainWindow):
    def __init__(self):
        super(DynamicChart, self).__init__()

        self._train_accuracy_x = 5
        self._train_accuracy_y = 1
        self._train_loss_x = 5
        self._train_loss_y = 1
        self._test_accuracy_x = 5
        self._test_accuracy_y = 1
        self._test_loss_x = 5
        self._test_loss_y = 1

        self._axisX = QValueAxis()
        self._axisX.setTickCount(21)
        self._axisX.setRange(-15, 5)
        self._axisX.setLabelFormat("%d")
        self._axisY = QValueAxis()
        self._axisY.setRange(0, 15)

        self.train_accuracy_series = QSplineSeries()
        self.train_accuracy_series.append(self._train_accuracy_x, self._train_accuracy_y)
        self.train_accuracy_series.setName("train_accuracy")

        self.train_loss_series = QSplineSeries()
        self.train_loss_series.append(self._train_loss_x, self._train_loss_y)
        self.train_loss_series.setName("train_loss")

        self.test_accuracy_series = QSplineSeries()
        self.test_accuracy_series.append(self._test_accuracy_x, self._test_accuracy_y)
        self.test_accuracy_series.setName("test_accuracy")

        self.test_loss_series = QSplineSeries()
        self.test_loss_series.append(self._test_loss_x, self._test_loss_y)
        self.test_loss_series.setName("test_loss")

        self.chart = QChart()
        self.chart.addSeries(self.train_accuracy_series)
        self.chart.addSeries(self.train_loss_series)
        self.chart.addSeries(self.test_accuracy_series)
        self.chart.addSeries(self.test_loss_series)

        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        self.chart.legend().setFont(QFont("Arial", 10))
        self.chart.legend().setColor(QColor("gray"))

        # self.chart.legend().hide()
        self.chart.addAxis(self._axisX, Qt.AlignBottom)
        self.chart.addAxis(self._axisY, Qt.AlignLeft)
        self.chart.setAnimationOptions(QChart.AllAnimations)
        self.chart.setTitle("Dynamic chart test")

        self.train_accuracy_series.attachAxis(self._axisX)
        self.train_accuracy_series.attachAxis(self._axisY)
        self.train_loss_series.attachAxis(self._axisX)
        self.train_loss_series.attachAxis(self._axisY)
        self.test_accuracy_series.attachAxis(self._axisX)
        self.test_accuracy_series.attachAxis(self._axisY)
        self.test_loss_series.attachAxis(self._axisX)
        self.test_loss_series.attachAxis(self._axisY)

        self._chart_view = QChartView(self.chart)
        self._chart_view.setRenderHint(QPainter.Antialiasing)

        self.setCentralWidget(self._chart_view)

    @Slot(float)
    def updateTrainAccuracy(self, value):
        y = (self._axisX.max() - self._axisX.min()) / self._axisX.tickCount()
        self._train_accuracy_x += y
        self._train_accuracy_y = value
        self.train_accuracy_series.append(self._train_accuracy_x, self._train_accuracy_y)

    @Slot(float)
    def updateTrainLoss(self, value):
        y = (self._axisX.max() - self._axisX.min()) / self._axisX.tickCount()
        self._train_loss_x += y
        self._train_loss_y = value
        self.train_loss_series.append(self._train_loss_x, self._train_loss_y)

    @Slot(float)
    def updateTestAccuracy(self, value):
        y = (self._axisX.max() - self._axisX.min()) / self._axisX.tickCount()
        self._test_accuracy_x += y
        self._test_accuracy_y = value
        self.test_accuracy_series.append(self._test_accuracy_x, self._test_accuracy_y)

    @Slot(float)
    def updateTestLoss(self, value):
        x = self.chart.plotArea().width() / self._axisX.tickCount()
        y = (self._axisX.max() - self._axisX.min()) / self._axisX.tickCount()
        self._test_loss_x += y
        self._test_loss_y = value
        self.test_loss_series.append(self._test_loss_x, self._test_loss_y)

        self.chart.scroll(x, 0)


class Trainer(QObject):
    trainAccuracyChanged = Signal(float)
    trainLossChanged = Signal(float)
    testAccuracyChanged = Signal(float)
    testLossChanged = Signal(float)

    def __init__(self):
        super().__init__()

    def train(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        data_loader = MiniDataLoader()

        train_total = data_loader.total("train")
        val_total = data_loader.total("test")

        inner_optimizer = optimizers.Adam(args.inner_lr)
        outer_optimizer = optimizers.Adam(args.outer_lr)

        maml = MAML(args.input_shape, args.n_way)
        # 验证次数可以少一些，不需要每次都更新这么多
        val_steps = 3
        train_steps = 1
        maml.meta_model.load_weights("maml_mini.h5")

        for e in range(args.epochs):

            train_progbar = utils.Progbar(train_steps)
            val_progbar = utils.Progbar(val_steps)
            print('\nEpoch {}/{}'.format(e + 1, args.epochs))

            train_meta_loss = []
            train_meta_acc = []
            val_meta_loss = []
            val_meta_acc = []

            for i in range(train_steps):
                batch_train_loss, acc = maml.train_on_batch(data_loader.sampleBatchDataset(args.batch_size),
                                                            inner_optimizer,
                                                            inner_step=15,
                                                            outer_optimizer=outer_optimizer)

                train_meta_loss.append(batch_train_loss)
                train_meta_acc.append(acc)
                train_progbar.update(i + 1, [('loss', np.mean(train_meta_loss)),
                                             ('accuracy', np.mean(train_meta_acc))])

            for i in range(val_steps):
                batch_val_loss, val_acc = maml.train_on_batch(
                    data_loader.sampleBatchDataset(args.batch_size, training=False),
                    inner_optimizer,
                    inner_step=15,
                    train_step=False, painting=True)
                # print(type(batch_val_loss))

                val_meta_loss.append(batch_val_loss)
                val_meta_acc.append(val_acc)
                val_progbar.update(i + 1, [('val_loss', np.mean(val_meta_loss)),
                                           ('val_accuracy', np.mean(val_meta_acc))])

            self.trainAccuracyChanged.emit(np.mean(train_meta_acc))
            self.trainLossChanged.emit(np.mean(train_meta_loss))
            self.testAccuracyChanged.emit(np.mean(val_meta_acc))
            self.testLossChanged.emit(np.mean(val_meta_loss))

            maml.meta_model.save_weights("maml_mini.h5")

if __name__ == "__main__":
    app = QApplication([])
    window = DynamicChart()
    window.show()
    window.resize(600, 300)

    trainer = Trainer()

    trainerThread = threading.Thread(target=trainer.train)

    trainerThread.start()

    trainer.trainAccuracyChanged.connect(window.updateTrainAccuracy)
    trainer.trainLossChanged.connect(window.updateTrainLoss)
    trainer.testAccuracyChanged.connect(window.updateTestAccuracy)
    trainer.testLossChanged.connect(window.updateTestLoss)

    sys.exit(app.exec())
