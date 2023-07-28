import pickle
import time
import os

from keras import optimizers, utils
import tensorflow as tf
import numpy as np

from dataReader import MAMLDataLoader
# from net import MAML
from net_multiStep import MAML
# from net_multiStep_secondDerivative import MAML
from config import args

train_data = MAMLDataLoader(args.train_data_dir, args.batch_size)
val_data = MAMLDataLoader(args.val_data_dir, args.val_batch_size)

# 余弦退火调整学习率
learning_rate_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=args.outer_lr, decay_steps=200, alpha=0.001)

inner_optimizer = optimizers.Adam(args.inner_lr)
outer_optimizer = optimizers.Adam(learning_rate_scheduler)

maml = MAML(args.input_shape, args.n_way)
train_data_steps = 20
val_data_steps = 5
train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list = [], [], [], []
total_time = 0
for e in range(args.epochs):
    start = time.time()
    train_progbar = utils.Progbar(train_data_steps)
    val_progbar = utils.Progbar(val_data_steps)
    print('\nEpoch {}/{}'.format(e + 1, args.epochs))

    train_meta_loss = []
    train_meta_acc = []
    val_meta_loss = []
    val_meta_acc = []

    for i in range(train_data_steps):
        batch_train_loss, acc = maml.train_on_batch(train_data.get_one_batch(),
                                                    inner_optimizer,
                                                    inner_step=15,
                                                    outer_optimizer=outer_optimizer,
                                                    training=True)

        train_meta_loss.append(batch_train_loss)
        train_meta_acc.append(acc)
        train_progbar.update(i + 1, [('loss', np.mean(train_meta_loss)),
                                     ('accuracy', np.mean(train_meta_acc))])

    for i in range(val_data_steps):
        batch_val_loss, val_acc = maml.train_on_batch(val_data.get_one_batch(),
                                                      inner_optimizer,
                                                      inner_step=15,
                                                      train_step=False,
                                                      training=False)

        val_meta_loss.append(batch_val_loss)
        val_meta_acc.append(val_acc)
        val_progbar.update(i + 1, [('val_loss', np.mean(val_meta_loss)),
                                   ('val_accuracy', np.mean(val_meta_acc))])

    # maml.meta_model.save_weights("maml_omniglot_5way_1shot.h5")
    train_loss_list.append(np.mean(train_meta_loss))
    train_accuracy_list.append(np.mean(train_meta_acc))
    test_loss_list.append(np.mean(val_meta_loss))
    test_accuracy_list.append(np.mean(val_meta_acc))

with open("matching_omniglot_5way_1shot.pkl", "wb") as f:
    pickle.dump((train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list), f)
