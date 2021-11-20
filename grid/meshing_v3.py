import numpy as np

# 二维结构化网格划分 第一列为z轴 第二列为ρ轴
# ********************************************纳米区域*********************************************
# 绘制纳米区域的z轴
nano_z1 = np.linspace(-7.5, -0.5, 8)  # 间距为 1nm 的部分
nano_z2 = np.linspace(-0.5, 7.5, 17)  # 间距为 0.5nm 的部分
nano_z = np.concatenate((nano_z1[:-1], nano_z2))

# 绘制纳米区域的ρ轴，也可称为r轴
nano_r1 = np.linspace(0, 15, 31)  # 间距为 0.5nm 的部分
nano_r2 = np.linspace(15, 50, 36)  # 间距为 1nm 的部分
nano_r3 = np.linspace(50, 100, 26)  # 间距为 2nm 的部分
nano_r4 = np.linspace(100, 296, 50)  # 间距为 4nm 的部分
nano_r = np.concatenate((nano_r1[:-1], nano_r2[:-1], nano_r3[:-1], nano_r4))

# 拼合出纳米区域各点坐标
nano_coordinates = np.concatenate(
    (np.tile(nano_z, len(nano_r))[:, np.newaxis], np.repeat(nano_r, len(nano_z))[:, np.newaxis]), axis=1)

# ********************************************开放区域*********************************************
# 绘制开放区域的z轴
open_z1 = np.insert(np.cumsum(np.arange(5, 71)), 0, 0)  # 正半轴
open_z2 = np.negative(np.flipud(open_z1[1:]))  # 负半轴
open_z = np.concatenate((open_z2, open_z1))

# 绘制开放区域的ρ轴，也可称为r轴
open_r = 300 + np.cumsum(np.arange(5, 97))

# 拼合出开放区域各点坐标
open_coordinates = np.concatenate(
    (np.tile(open_z, len(open_r))[:, np.newaxis], np.repeat(open_r, len(open_z))[:, np.newaxis]), axis=1)

# ********************************************交界处，即即 r=300 处*********************************************
boundary_z = np.unique(np.concatenate((nano_z, open_z)))
boundary_coordinates = np.concatenate(
    (boundary_z[:, np.newaxis], np.full(len(boundary_z), 300, float)[:, np.newaxis]), axis=1)

# ********************************************得到整个网格的点坐标*******************************************************
grid_coordinates = np.concatenate((nano_coordinates, boundary_coordinates, open_coordinates))
np.savetxt("grid_coordinates.csv", grid_coordinates, fmt='%.1f', delimiter=",")

# 统计信息
counts = np.asarray([len(nano_coordinates), len(boundary_coordinates), len(open_coordinates), len(grid_coordinates)])
print("纳米区域点总数：{0[0]}\t索引：0-{1}\n"
      "交界处点总数：{0[1]}\t索引：{0[0]}-{2}\n"
      "开放区域点总数：{0[2]}\t索引：{3}-{4}\n"
      "点总数：{0[3]}\n".
      format(counts, counts[0] - 1, counts[0] + counts[1] - 1, counts[0] + counts[1], counts[3] - 1))
