import os
from math import sqrt, log, exp, pi
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import convolve1d

global old_coordinates_df, old_coordinates, CCAF


# *************************************  输入数据  ****************************************

def load_cylinder_coordinates():
    """
    加载柱坐标系
    """
    global old_coordinates_df, old_coordinates
    old_coordinates_df = pd.read_csv("grid/grid_coordinates.csv", header=None)
    old_coordinates = np.array(old_coordinates_df)


def load_c_caf(file):
    """
    加载浓度值文件
    """
    global CCAF
    caf_df = pd.read_csv("data/" + file, header=None)
    CCAF = np.array(caf_df.iloc[:, 0]).astype('float64')


def simplify_cylinder_coordinates():
    """
    简化柱坐标系，方便搜索对应点浓度，存储为polar_concentration，大小(16,141)

    :return: 简化后的柱坐标对应的浓度以及柱坐标半径长
    """
    nano_r1 = np.linspace(0, 15, 31)  # 间距为 0.5nm 的部分
    nano_r2 = np.linspace(15, 50, 36)  # 间距为 1nm 的部分
    nano_r3 = np.linspace(50, 100, 26)  # 间距为 2nm 的部分
    nano_r4 = np.linspace(100, 300, 51)  # 间距为 4nm 的部分
    nano_r = np.concatenate((nano_r1[:-1], nano_r2[:-1], nano_r3[:-1], nano_r4))

    nano_z1 = np.linspace(7.5, 0.5, 8)  # 间距为 1nm 的部分，舍去间距为0.5nm
    nano_z2 = np.linspace(-0.5, -7.5, 8)  # 间距为 1nm 的部分
    nano_z = np.concatenate((nano_z1, nano_z2))

    polar_concentration = np.zeros((len(nano_z), len(nano_r)))
    for i in range(0, 16):
        xy_dataFrame = old_coordinates_df.loc[old_coordinates_df[0] == nano_z[i]]
        xy_dataFrame.columns = ['z', 'r']
        for j in range(len(nano_r)):
            index = xy_dataFrame[xy_dataFrame.r == nano_r[j]].iloc[0, :].name
            polar_concentration[i][j] = CCAF[index]

    return polar_concentration, nano_r


def quarter_cube_c_caf(polar_concentration, nano_r):
    """
    计算直角坐标系x,y正半轴,z轴全部的1/4立方体的浓度

    :param polar_concentration:简化后的柱坐标对应的浓度
    :param nano_r:柱坐标半径长
    :return: 1/4立方体的浓度
    """
    quarter_c_caf = np.zeros((16, 301, 301))
    for i in range(0, 301):
        for j in range(0, i + 1):
            radius = sqrt(i ** 2 + j ** 2)
            # searchsorted:查找下标，找不到就用前一个下标+1
            index = np.searchsorted(nano_r, radius)
            if index >= len(nano_r):
                # 超出半径300
                for k in range(0, 16):
                    quarter_c_caf[k][i][j] = 0.0
                    quarter_c_caf[k][j][i] = 0.0
            elif nano_r[index] == radius:
                # 正好对应柱坐标系
                for k in range(0, 16):
                    quarter_c_caf[k][i][j] = polar_concentration[k][index]
                    quarter_c_caf[k][j][i] = polar_concentration[k][index]
            else:
                # 不在柱坐标系上，需要线性插值
                pre_radius = nano_r[index - 1]
                next_radius = nano_r[index]
                ratio = (radius - pre_radius) / (next_radius - pre_radius)
                for k in range(0, 16):
                    pre_caf = polar_concentration[k][index - 1]
                    next_caf = polar_concentration[k][index]
                    caf = next_caf * ratio + (1 - ratio) * pre_caf
                    quarter_c_caf[k][i][j] = caf
                    quarter_c_caf[k][j][i] = caf
    return quarter_c_caf


def flip_and_concatenate(i, quarter_c_caf):
    """
    将x,y正半轴的正方形（二维数组），分别沿x轴和y轴翻转拼接成一整个正方形
    相当于一层

    :param i: 第i层
    :param quarter_c_caf:1/4立方体的浓度
    :return: 第i层通过翻转拼接得到的大正方形的浓度
    """
    plane = quarter_c_caf[i]
    column = plane[1:, 0]
    row = plane[0, 1:]
    rest = plane[1:, 1:]

    column_flip = np.flip(column, axis=0)
    row_flip = np.flip(row, axis=0)

    rest_ud_flip = np.flip(rest, axis=0)
    rest_lr_flip = np.flip(rest, axis=1)
    rest_ud_lr_flip = np.flip(rest_ud_flip, axis=1)

    column_flip = column_flip[:, np.newaxis]  # 一维转二维
    row_flip = row_flip[np.newaxis, :]

    part1 = np.concatenate((rest_ud_lr_flip, column_flip, rest_ud_flip), axis=1)
    part2 = np.concatenate((row_flip, rest_lr_flip), axis=0)
    part3 = np.concatenate((part2, plane), axis=1)
    c_caf_2d = np.concatenate((part1, part3), axis=0)

    return c_caf_2d


def total_cube_c_caf(quarter_c_caf):
    """
    拼接成整个立方体

    :param quarter_c_caf: 1/4立方体的浓度
    :return: 完整的立方体的浓度，也就是卷积过程的输入数据
    """

    # 从第0层开始（对应z轴为0）
    a = flip_and_concatenate(0, quarter_c_caf)
    a = a[np.newaxis, :]
    # 将第i层与前面数据累加
    for i in range(1, 16):
        b = flip_and_concatenate(i, quarter_c_caf)
        b = b[np.newaxis, :]
        a = np.vstack((a, b))
    total_c_caf = a
    return total_c_caf


# ************************************  卷积核 ****************************************
def kernel_psf():
    """
    计算卷积核，将卷积核划分为两个较小的卷积核
    """
    xy_psf = np.zeros(700)
    z_psf = np.zeros(700)

    # xy_psf:518 z_psf:1034
    # 400 800

    coe_f = 1.0 / (sqrt(0.25 * pi / log(2.0)) * 542)
    sigma = 542 ** 2 / log(2.0) * 0.25
    for i in range(len(xy_psf)):
        xy_psf[i] = coe_f * exp(-i ** 2 / sigma)

    coe_f = 1.0 / (sqrt(0.25 * pi / log(2.0)) * 542)
    sigma = 542 ** 2 / log(2.0) * 0.25
    for i in range(len(z_psf)):
        z_psf[i] = coe_f * exp(-i ** 2 / sigma)

    xy_psf_flip = np.flip(xy_psf[1:], axis=0)
    xy_psf = np.concatenate((xy_psf_flip, xy_psf), axis=0)
    z_psf_flip = np.flip(z_psf[1:], axis=0)
    z_psf = np.concatenate((z_psf_flip, z_psf), axis=0)

    return xy_psf, z_psf


# ************************************  卷积  ****************************************
def convolve(c_caf, xy_psf, z_psf):
    """
    空间可分离卷积
    """
    # 对x轴作一维卷积
    for i in range(0, 16):
        for k in range(0, 601):
            x_data = np.array(c_caf[i, :, k])
            convolve_with_x_psf = convolve1d(x_data, xy_psf, mode='constant', cval=0)
            c_caf[i, :, k] = convolve_with_x_psf
    # 对y轴作一维卷积
    for i in range(0, 16):
        for j in range(0, 601):
            y_data = np.array(c_caf[i, j, :])
            convolve_with_y_psf = convolve1d(y_data, xy_psf, mode='constant', cval=0)
            c_caf[i, j, :] = convolve_with_y_psf
    # 对z轴作一维卷积
    for j in range(0, 601):
        for k in range(0, 601):
            z_data = np.array(c_caf[:, j, k])
            convolve_with_z_psf = convolve1d(z_data, z_psf, mode='constant', cval=0)
            c_caf[:, j, k] = convolve_with_z_psf
    return c_caf


def plot(c_caf, file):
    x = []
    y = []
    for i in range(0, 601):
        x.append(i - 300)
        y.append(c_caf[0][300][i])
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.ticklabel_format(style='plain')
    plt.grid()
    plt.savefig("figure/convolved_new_" + os.path.splitext(file)[0] + ".jpg")
    plt.show()


if __name__ == '__main__':
    # 先加载柱坐标系坐标以及各点对应的浓度，并简化
    file = "CaF00010000.csv"
    load_cylinder_coordinates()
    load_c_caf(file)
    polar_concentration, nano_r = simplify_cylinder_coordinates()
    # 输入数据
    quarter_c_caf = quarter_cube_c_caf(polar_concentration, nano_r)
    total_c_caf = total_cube_c_caf(quarter_c_caf)
    # 卷积核
    xy_psf, z_psf = kernel_psf()
    # 卷积
    c_caf_convolved = convolve(total_c_caf, xy_psf, z_psf)
    # 绘图
    plot(c_caf_convolved, file)
