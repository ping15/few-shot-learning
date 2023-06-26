import pickle

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
with open("origin.pkl", "rb") as f:
    origin_loss, origin_accuracy, origin_val_loss, origin_val_accuracy = pickle.load(f)

with open("origin_model.pkl", "rb") as f:
    origin_model_loss, origin_model_accuracy, origin_model_val_loss, origin_model_val_accuracy = pickle.load(f)

with open("terminal.pkl", "rb") as f:
    terminal_loss, terminal_accuracy, terminal_val_loss, terminal_val_accuracy = pickle.load(f)

with open("origin_model_innerCount.pkl", "rb") as f:
    origin_model_innerCount_loss, origin_model_innerCount_accuracy, origin_model_innerCount_val_loss, origin_model_innerCount_val_accuracy = pickle.load(f)

# 训练损失
plt.figure(figsize=(16, 16))
plt.plot(origin_loss)
plt.plot(origin_model_loss)
plt.plot(origin_model_innerCount_loss)
plt.plot(terminal_loss)
plt.xlabel("迭代次数")
plt.ylabel("损失")
plt.legend(["原始", "原始+He参数初始化", "原始+He参数初始化+增大内部循环次数", "原始+He参数初始化+增大内部循环次数+MAML测试集内部循环时将数据打乱"], loc='best')
# plt.title("MAML在训练时的损失")
# plt.figtext(0.5, 0.01, "图4.1 MAML在训练和测试时的损失", ha='center', fontsize=16)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.show()

# 训练准确率
plt.figure(figsize=(16, 16))
plt.plot(origin_accuracy)
plt.plot(origin_model_accuracy)
plt.plot(origin_model_innerCount_accuracy)
plt.plot(terminal_accuracy)
plt.xlabel("迭代次数")
plt.ylabel("准确率")
plt.legend(["原始", "原始+He参数初始化", "原始+He参数初始化+增大内部循环次数", "原始+He参数初始化+增大内部循环次数+MAML测试集内部循环时将数据打乱"], loc='best')
# plt.title("MAML在训练时的准确率")
# plt.figtext(0.5, 0.01, "图4.1 MAML在训练和测试时的准确率", ha='center', fontsize=16)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.show()

# 测试损失
plt.figure(figsize=(16, 16))
plt.plot(origin_val_loss)
plt.plot(origin_model_val_loss)
plt.plot(origin_model_innerCount_val_loss)
plt.plot(terminal_val_loss)
plt.xlabel("迭代次数")
plt.ylabel("损失")
plt.legend(["原始", "原始+He参数初始化", "原始+He参数初始化+增大内部循环次数", "原始+He参数初始化+增大内部循环次数+MAML测试集内部循环时将数据打乱"], loc='best')
# plt.title("MAML在训练时的损失")
# plt.figtext(0.5, 0.01, "图4.1 MAML在训练和测试时的损失", ha='center', fontsize=16)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.show()

# 测试准确率
plt.figure(figsize=(16, 16))
plt.plot(origin_val_accuracy)
plt.plot(origin_model_val_accuracy)
plt.plot(origin_model_innerCount_val_accuracy)
plt.plot(terminal_val_accuracy)
plt.xlabel("迭代次数")
plt.ylabel("准确率")
plt.legend(["原始", "原始+He参数初始化", "原始+He参数初始化+增大内部循环次数", "原始+He参数初始化+增大内部循环次数+MAML测试集内部循环时将数据打乱"], loc='best')
# plt.title("MAML在训练时的准确率")
# plt.figtext(0.5, 0.01, "图4.1 MAML在训练和测试时的准确率", ha='center', fontsize=16)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.show()
