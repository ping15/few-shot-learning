import numpy as np
import matplotlib.pyplot as plt

# 假设有10个点，x和y坐标都相同
num_points = 2
x = np.zeros(num_points)
y = np.zeros(num_points)

# 生成随机半径
r = np.random.uniform(0, 1, size=num_points)

# 生成随机角度
theta = np.random.uniform(0, 2*np.pi, size=num_points)

# 将极坐标转换为直角坐标
x = r * np.cos(theta)
y = r * np.sin(theta)

# 将x和y坐标平移，使得它们的均值为0
x = x - np.mean(x)
y = y - np.mean(y)

# 将x和y坐标缩放，使得它们的标准差为1
x = x / np.std(x)
y = y / np.std(y)

# 将x和y坐标缩放，使得它们的范围在-0.1到0.1之间
x = x * 0.1
y = y * 0.1

# 绘制散点图
plt.scatter(x, y)

# 显示图形
plt.show()
