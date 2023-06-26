import os

from keras import optimizers, utils
import numpy as np
import tensorflow as tf

from dataLoader.miniDataLoader import MiniDataLoader
from net_mini import MAML
from config_mini import args

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
val_steps = 10
train_steps = 20
# maml.meta_model.load_weights("maml_mini.h5")

for e in range(args.epochs):
    train_progbar = utils.Progbar(train_steps)
    val_progbar = utils.Progbar(val_steps)
    print('\nEpoch {}/{}'.format(e + 1, args.epochs))

    train_meta_loss = []
    train_meta_acc = []
    val_meta_loss = []
    val_meta_acc = []

    for i in range(train_steps):
        batch_train_loss, acc = maml.train_on_batch(
            data_loader.sampleBatchDataset(args.batch_size),
            inner_optimizer,
            inner_step=5,
            outer_optimizer=outer_optimizer)

        train_meta_loss.append(batch_train_loss)
        train_meta_acc.append(acc)
        train_progbar.update(i + 1, [('loss', np.mean(train_meta_loss)),
                                     ('accuracy', np.mean(train_meta_acc))])

    for i in range(val_steps):
        batch_val_loss, val_acc = maml.train_on_batch(
            data_loader.sampleBatchDataset(args.batch_size, training=False),
            inner_optimizer,
            inner_step=5,
            train_step=False,
            painting=False)
        # print(type(batch_val_loss))

        val_meta_loss.append(batch_val_loss)
        val_meta_acc.append(val_acc)
        val_progbar.update(i + 1, [('val_loss', np.mean(val_meta_loss)),
                                   ('val_accuracy', np.mean(val_meta_acc))])

    maml.meta_model.save_weights("maml_mini.h5")
