import re
import pickle

# 定义四个空列表
loss = []
accuracy = []
val_loss = []
val_accuracy = []

# 读取包含训练记录的文件
with open('data.log', 'r') as f:
    # 遍历每一行
    for line in f:
        # 使用正则表达式匹配数字并添加到相应的列表中
        if 'loss' in line and "val_loss" not in line:
            loss.append(float(re.findall(r'loss: (\d+\.\d+)', line)[0]))

        if 'accuracy' in line and "val_accuracy" not in line:
            accuracy.append(float(re.findall(r'accuracy: (\d+\.\d+)', line)[0]))

        if 'val_loss' in line:
            val_loss.append(float(re.findall(r'val_loss: (\d+\.\d+)', line)[0]))

        if 'val_accuracy' in line:
            val_accuracy.append(float(re.findall(r'val_accuracy: (\d+\.\d+)', line)[0]))

with open("data.pkl", "wb") as f:
    pickle.dump((loss, accuracy, val_loss, val_accuracy), f)
