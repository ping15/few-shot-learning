import random
import math

import numpy as np
from scipy.stats import t


def get_random_number():
    mu, sigma = 0, 0.1 # 均值和标准差
    x = []
    while len(x) < 1:
        r = random.uniform(0, 1) # 生成0到1之间的随机数
        y = math.sqrt(-2 * math.log(r)) * math.cos(2 * math.pi * r) # 使用Box-Muller变换生成正态分布随机数
        y = y * sigma + mu # 调整均值和标准差
        if -0.2 <= y <= 0.2: # 限制随机数在-0.1到0.1之间
            x.append(y)
    return x[0]


def get_t_random_number():
    # 设置自由度和均值
    df = 10
    mean = 0

    # 生成一个t分布的随机数
    x = t.rvs(df, loc=mean)

    # 缩放和平移
    x_scaled = (x - mean) / np.sqrt(df / (df - 2))
    x_shifted = x_scaled * 0.1

    # 将结果限制在-0.1到0.1的范围内
    if x_shifted < -0.1:
        x_shifted = -0.1
    elif x_shifted > 0.1:
        x_shifted = 0.1

    return x_shifted


def get_poly_random_number(num_points):
    # 假设有10个点，x和y坐标都相同
    x = np.zeros(num_points)
    y = np.zeros(num_points)

    # 生成随机半径
    r = np.random.uniform(0, 1, size=num_points)

    # 生成随机角度
    theta = np.random.uniform(0, 2 * np.pi, size=num_points)

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

    return x, y


if __name__ == "__main__":
    print(get_poly_random_number(1000))
