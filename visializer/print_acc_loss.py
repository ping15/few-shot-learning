import pickle

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# with open("loss_acc_data.pkl", "rb") as f:
#     loss, accuracy, val_loss, val_accuracy = pickle.load(f)
with open("matching_omniglot_5way_1shot.pkl", "rb") as f:
    loss, accuracy, val_loss, val_accuracy = pickle.load(f)

# 训练损失
plt.figure(figsize=(16, 16))
plt.plot(loss)
plt.plot(val_loss)
plt.xlabel("迭代次数")
plt.ylabel("损失")
plt.legend(["训练损失", "测试损失"], loc='best')
# plt.title("MAML在训练时的损失")
# plt.figtext(0.5, 0.01, "图4.1 MAML在训练和测试时的损失", ha='center', fontsize=16)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.show()

# 训练准确率
plt.figure(figsize=(16, 16))
plt.plot(accuracy)
plt.plot(val_accuracy)
plt.xlabel("迭代次数")
plt.ylabel("准确率")
plt.legend(["训练准确率", "测试准确率"], loc='best')
# plt.title("MAML在训练时的准确率")
# plt.figtext(0.5, 0.01, "图4.1 MAML在训练和测试时的准确率", ha='center', fontsize=16)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.show()
