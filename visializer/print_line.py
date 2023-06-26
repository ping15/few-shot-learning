import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
# 构造数据
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y1 = [92.41, 94.21, 95.45, 96.49, 96.97, 97.41, 97.44, 98.00, 98.14, 98.40]
y2 = [96.30, 97.41, 97.91, 98.11, 99.46, 99.65, 99.71, 99.70, 99.82, 99.78]
y3 = [93.25, 95.51, 97.11, 97.41, 98.15, 98.55, 98.81, 99.14, 99.41, 99.35]

# 画折线图1，设置颜色为蓝色，标记为圆圈
plt.plot(x, y1, color='blue', marker='o', label='MAML网络')

# 画折线图2，设置颜色为红色，标记为三角形
plt.plot(x, y2, color='red', marker='^', label='关系网络')

# 画折线图2，设置颜色为红色，标记为x
plt.plot(x, y3, color='green', marker='x', label='原型网络')

plt.xlabel("训练时shot的个数")
plt.ylabel("测试准确率(5-way)")

plt.title("omniglot")

# 添加网格
plt.grid(True)

# 添加图例
plt.legend()

plt.subplot(1, 2, 2)
# 构造数据
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y1 = [42.41, 43.21, 43.25, 44.49, 45.97, 46.41, 47.44, 48.00, 48.14, 48.40]
y2 = [45.41, 47.41, 47.91, 48.11, 48.41, 48.85, 49.01, 49.14, 49.41, 49.75]
y3 = [43.21, 43.41, 45.61, 46.11, 46.41, 46.85, 47.87, 48.54, 48.41, 48.75]

# 画折线图1，设置颜色为蓝色，标记为圆圈
plt.plot(x, y1, color='blue', marker='o', label='MAML网络')

# 画折线图2，设置颜色为红色，标记为三角形
plt.plot(x, y2, color='red', marker='^', label='关系网络')

# 画折线图2，设置颜色为红色，标记为x
plt.plot(x, y3, color='green', marker='x', label='原型网络')

plt.xlabel("训练时shot的个数")
plt.ylabel("测试准确率(5-way)")

plt.title("miniImageNet")

# 添加网格
plt.grid(True)

# 添加图例
plt.legend()

# plt.figtext(0.5, 0.005, "Figure(2-11)", ha='center', fontsize=16)

# 显示图形
plt.show()
