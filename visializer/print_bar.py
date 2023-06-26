import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义两组数据
x1 = [1, 2, 3, 4]
y1 = [92.34, 94.21, 81.77, 90.45]
x2 = [1.2, 2.2, 3.2, 4.2]
y2 = [96.30, 99.46, 92.88, 98.09]
x3 = [1.4, 2.4, 3.4, 4.4]
y3 = [93.25, 98.15, 89.14, 96.66]

# 绘制两组数据的柱状图
plt.bar(x1, y1, width=0.2, align='center', label='MAML网络', color='b', alpha=0.5)
plt.bar(x2, y2, width=0.2, align='center', label='关系网络', color='g', alpha=0.5)
plt.bar(x3, y3, width=0.2, align='center', label='原型网络', color='r', alpha=0.5)

# 调整两组数据的位置
for i in range(len(x1)):
    x1[i] -= 0.2
    x2[i] += 0
    x3[i] += 0.2

# 添加标签和标题
# plt.xlabel('X Label')
plt.ylabel('准确率')
# plt.figtext(0.5, 0.01, "Figure(2-10)", ha='center', fontsize=16)
# plt.title('MAML和关系网络在不同的way和shot情况下的表现')
plt.xticks([1, 2, 3, 4], ['5-way 1-shot', '5-way 5-shot', '20-way 1-shot', '20-way 5-shot'])

# 隐藏右边和上边的坐标轴边框
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# 添加图例
plt.legend(loc='lower center')

# 将 y 轴数值转换为字符串格式并添加百分号
fmt = '%.0f%%'
yticks = mtick.FormatStrFormatter(fmt)
ax.yaxis.set_major_formatter(yticks)

# 显示图形
plt.show()
