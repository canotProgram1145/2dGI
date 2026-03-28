
from typing import List

import math
import numpy as np
from typing import List

def sdf_2d(f: List[float]) -> List[float]:
    """
    一维精确欧氏距离变换（支持权重）
    f: 一维数组，f[i] 表示在位置 i 的权重（抛物线最低点）。
       对于种子点，f[i] = 0；对于非种子点，应设为 inf（会被忽略）。
    """
    n = len(f)
    v = [0] * n          # 抛物线顶点索引
    z = [0.0] * (n + 1)  # 相邻抛物线的交点横坐标
    k = 0                # 当前下包络中的抛物线数量

    # 构建下包络
    for q in range(n):
        if math.isinf(f[q]):
            continue
        # 当前抛物线: (x - q)^2 + f[q]
        while k > 0:
            # 计算当前抛物线 v[k-1] 和 q 的交点
            delta = (f[q] - f[v[k-1]]) / (q - v[k-1])
            s = (q + v[k-1] + delta) / 2.0
            if s > z[k-1]:
                break
            k -= 1
        v[k] = q
        if k > 0:
            delta = (f[q] - f[v[k-1]]) / (q - v[k-1])
            z[k] = (q + v[k-1] + delta) / 2.0
        else:
            z[k] = -float('inf')
        k += 1

    # 如果没有有效抛物线（即所有 f 均为 inf）
    if k == 0:
        return [float('inf')] * n

    z[k] = float('inf')  # 哨兵

    # 计算每个位置的最小值
    d = [0.0] * n
    k = 0
    for x in range(n):
        while k + 1 <= n and z[k + 1] < x:
            k += 1
        dx = x - v[k]
        d[x] = dx * dx + f[v[k]]
    return d

def edt_2d_unsigned(bool_array: List[List[bool]]) -> List[List[float]]:
    """
    二维精确欧氏距离变换（无符号），返回每个像素到最近前景点的欧氏距离。
    bool_array: 二维列表，True 表示前景点，False 表示背景点。
    """
    if not bool_array or not bool_array[0]:
        return []
    height = len(bool_array)
    width = len(bool_array[0])

    INF = float('inf')

    # 第一次：行变换
    g = [[INF] * width for _ in range(height)]
    for i in range(height):
        row = [0.0 if bool_array[i][j] else INF for j in range(width)]
        dist_sq = edt_1d(row)
        for j in range(width):
            g[i][j] = dist_sq[j]

    # 第二次：列变换
    result = [[0.0] * width for _ in range(height)]
    for j in range(width):
        col = [g[i][j] for i in range(height)]
        dist_sq = edt_1d(col)
        for i in range(height):
            result[i][j] = math.sqrt(dist_sq[i])
    return result

def sdf_2d(bool_array: List[List[bool]]) -> List[List[float]]:
    """
    生成有符号距离场（SDF）。
    bool_array: True 表示物体内部（前景），False 表示外部（背景）。
    返回二维浮点数列表，内部为负，外部为正，表面为0。
    """
    height = len(bool_array)
    if height == 0:
        return []
    width = len(bool_array[0])

    # 计算到最近前景的距离（背景点得到正值，前景点为0）
    dist_to_foreground = edt_2d_unsigned(bool_array)

    # 计算到最近背景的距离（前景点得到正值，背景点为0）
    # 通过反转布尔数组实现
    inverted = [[not val for val in row] for row in bool_array]
    dist_to_background = edt_2d_unsigned(inverted)

    # 组合成有符号距离场
    sdf = [[0.0] * width for _ in range(height)]
    for i in range(height):
        for j in range(width):
            if bool_array[i][j]:
                # 前景内部：距离为负（到背景的距离）
                # 如果背景不存在（dist_to_background 为 inf），则设为负无穷
                if math.isinf(dist_to_background[i][j]):
                    sdf[i][j] = -float('inf')
                else:
                    sdf[i][j] = -dist_to_background[i][j]
            else:
                # 背景外部：距离为正（到前景的距离）
                if math.isinf(dist_to_foreground[i][j]):
                    sdf[i][j] = float('inf')
                else:
                    sdf[i][j] = dist_to_foreground[i][j]
    return sdf


def edt_1d(f: List[float]) -> List[float]:
    """
    一维精确欧氏距离变换（支持权重）
    f: 一维数组，f[i] 表示在位置 i 的权重（可以看作是抛物线的最低点）。
       对于种子点，f[i] = 0；对于非种子点，应设为 inf（会被忽略）。
    """
    n = len(f)
    v = [0] * n          # 抛物线顶点索引
    z = [0.0] * (n + 1)  # 相邻抛物线的交点横坐标
    k = 0                # 当前下包络中的抛物线数量

    # 构建下包络
    for q in range(n):
        if math.isinf(f[q]):
            continue
        # 当前抛物线: (x - q)^2 + f[q]
        while k > 0:
            # 计算当前抛物线 v[k-1] 和 q 的交点
            delta = (f[q] - f[v[k-1]]) / (q - v[k-1])
            s = (q + v[k-1] + delta) / 2.0
            if s > z[k-1]:
                break
            k -= 1
        v[k] = q
        if k > 0:
            delta = (f[q] - f[v[k-1]]) / (q - v[k-1])
            z[k] = (q + v[k-1] + delta) / 2.0
        else:
            z[k] = -float('inf')
        k += 1

    # 如果没有有效抛物线（即所有 f 均为 inf）
    if k == 0:
        return [float('inf')] * n

    z[k] = float('inf')  # 哨兵

    # 计算每个位置的最小值
    d = [0.0] * n
    k = 0
    for x in range(n):
        while k + 1 <= n and z[k + 1] < x:
            k += 1
        dx = x - v[k]
        d[x] = dx * dx + f[v[k]]
    return d

def edt_2d(bool_array: List[List[bool]]) -> List[List[float]]:
    """
    二维精确欧氏距离变换（返回欧氏距离）
    bool_array: 二维列表，True 表示前景点，False 表示背景点
    """
    if not bool_array or not bool_array[0]:
        return []
    height = len(bool_array)
    width = len(bool_array[0])

    INF = float('inf')

    # 第一次：行变换
    g = [[INF] * width for _ in range(height)]
    for i in range(height):
        # 构建该行的权重数组：种子点为 0，非种子点为 inf
        row = [0.0 if bool_array[i][j] else INF for j in range(width)]
        dist_sq = edt_1d(row)
        for j in range(width):
            g[i][j] = dist_sq[j]

    # 第二次：列变换
    result = [[0.0] * width for _ in range(height)]
    for j in range(width):
        col = [g[i][j] for i in range(height)]
        dist_sq = edt_1d(col)
        for i in range(height):
            result[i][j] = math.sqrt(dist_sq[i])  # 开平方得到欧氏距离
    return result

# ------------------- 测试 -------------------
if __name__ == "__main__":
    test = [
        [False, False, False],
        [False, True,  False],
        [False, False, False]
    ]
    dist = sdf_2d(test)
    for row in dist:
        print([f"{v:.2f}" for v in row])