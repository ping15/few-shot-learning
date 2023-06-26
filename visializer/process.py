import pickle
import numpy as np

train_meta_loss = []
train_meta_acc = []
val_meta_loss = []
val_meta_acc = []
with open("terminal.pkl", "rb") as f:
    loss_list, accuracy_list, val_loss_list, val_accuracy_list = pickle.load(f)
    for loss in loss_list:
        train_meta_loss.append(np.mean(loss))

    for accuracy in accuracy_list:
        train_meta_acc.append(np.mean(accuracy))

    for val_loss in val_loss_list:
        val_meta_loss.append(np.mean(val_loss))

    for val_accuracy in val_accuracy_list:
        val_meta_acc.append(np.mean(val_accuracy))

with open("terminal.pkl", "wb") as f:
    pickle.dump((train_meta_loss, train_meta_acc, val_meta_loss, val_meta_acc), f)

