import numpy as np

def bilinear_interpolate(array, x, y, fill_value=None):
    """
    对二维网格数据进行双线性插值。

    参数：
        array : 二维数组（形状 (rows, cols)），网格点上的值，坐标 (i, j) 对应 array[i][j]。
        x : float，插值点的列坐标（对应列索引）。
        y : float，插值点的行坐标（对应行索引）。
        fill_value : float, optional，当坐标超出网格范围时使用的填充值。
                     若为 None，则返回最近边界点的值。

    返回：
        float : 插值结果。
    """
    # 将输入转换为 NumPy 数组（便于处理）
    arr = np.asarray(array)
    rows, cols = arr.shape

    # 处理坐标超出范围的情况
    if fill_value is not None:
        if x < 0 or x > cols - 1 or y < 0 or y > rows - 1:
            return fill_value
    else:
        # 钳位到边界内
        x = max(0, min(x, cols - 1))
        y = max(0, min(y, rows - 1))

    # 获取四个相邻网格点的索引
    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1

    # 处理边界情况（当坐标正好在边界上时，确保索引不越界）
    if x1 >= cols:
        x1 = x0
    if y1 >= rows:
        y1 = y0

    # 如果点正好落在网格点上，直接返回
    if x0 == x1 and y0 == y1:
        return arr[y0, x0]

    # 四个角点的值
    Q11 = arr[y0, x0]
    Q12 = arr[y1, x0]
    Q21 = arr[y0, x1]
    Q22 = arr[y1, x1]

    # x 方向权重
    if x1 == x0:
        wx = 0.0
    else:
        wx = (x - x0) / (x1 - x0)

    # y 方向权重
    if y1 == y0:
        wy = 0.0
    else:
        wy = (y - y0) / (y1 - y0)

    # 在 x 方向插值
    f_x_y0 = (1 - wx) * Q11 + wx * Q21
    f_x_y1 = (1 - wx) * Q12 + wx * Q22

    # 在 y 方向插值
    result = (1 - wy) * f_x_y0 + wy * f_x_y1

    return result

# 示例用法
if __name__ == "__main__":
    # 示例二维数组（4x4）
    data = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]

    # 插值点 (x=2.3, y=1.7)
    value = bilinear_interpolate(data, 2.3, 1.7)
    print(f"插值结果: {value}")  # 应接近 8.7（手动计算验证）