
from PIL import Image
#from numba.core.typing.builtins import Print
#from PIL import paste
from PIL import Image, ImageChops



import time
from numba import njit, prange

import math
import random
import numpy as np


random.seed(time.time())
from typing import List

cache=True
"""
class ray:
    O=np.array([0,0])
    P=np.array([0,0])
    def __init__(self,og,d):
        self.o=og
        self.d = d

    def ray(self):
        return self.o,self.d

"""

# ---------- 辅助函数 ----------



@njit
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



@njit
def rayp(ra, t):
    return ra[0] + t * ra[1]

def addlist(a, b):
    return [a[i] + b[i] for i in range(len(a))]

@njit
def re(a, b):
    r = int(a[0]) * int(b[0]) // 255  # 使用整数除法 // 替代 math.floor
    g = int(a[1]) * int(b[1]) // 255
    b_ = int(a[2]) * int(b[2]) // 255
    return np.array([r, g, b_, 255], dtype=np.int64)


@njit
def normal_at(sdf, x, y, epsilon=1e-4):
    """返回 (x,y) 处的单位法线（指向外部）"""
    dx = (bilinear_interpolate(sdf, x + epsilon, y) -
          bilinear_interpolate(sdf, x - epsilon, y))
    dy = (bilinear_interpolate(sdf, x, y + epsilon) -
          bilinear_interpolate(sdf, x, y - epsilon))
    grad = np.array([dx, dy])
    norm = np.linalg.norm(grad)
    if norm == 0:
        return np.array([0.0, 0.0])
    return grad / norm



@njit
def find_surface(ra, t_out, t_in, SDF, max_iter=100, tol=1e-7):
    """在 t_out (外部) 和 t_in (内部) 之间二分查找表面点"""
    for _ in range(max_iter):
        t_mid = (t_out + t_in) * 0.5
        pos_mid = rayp(ra, t_mid)
        d = bilinear_interpolate(SDF, pos_mid[0], pos_mid[1])
        if abs(d) < tol or (t_in - t_out) < tol:
            return t_mid, pos_mid
        if d > 0:  # 在外部
            t_out = t_mid
        else:      # 在内部
            t_in = t_mid
    # 返回最后的中点
    t_mid = (t_out + t_in) * 0.5
    return t_mid, rayp(ra, t_mid)





def srgb_to_linear_rgba(img):
    """
    转换 RGBA 图像：RGB 通道做 sRGB→线性，Alpha 通道保持不变
    img: uint8 array [H,W,4] 或 [...,4]
    返回: uint8 array 相同形状
    """
    if img.shape[-1] != 4:
        raise ValueError("需要 RGBA 数组，最后一维为 4")

    # 分离 RGB 和 Alpha
    rgb = img[..., :3]
    alpha = img[..., 3:]

    # 转换 RGB
    v = rgb.astype(np.float32) / 255.0
    mask = v <= 0.04045
    lin = np.where(mask, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    rgb_linear = (lin * 255.0 + 0.5).astype(np.uint8)

    # 合并结果
    return np.concatenate([rgb_linear, alpha], axis=-1)


def linear_to_srgb_rgba(img):
    """
    转换 RGBA 图像：RGB 通道做线性→sRGB，Alpha 通道保持不变
    img: uint8 array [H,W,4] 或 [...,4]
    返回: uint8 array 相同形状
    """
    if img.shape[-1] != 4:
        raise ValueError("需要 RGBA 数组，最后一维为 4")

    # 分离 RGB 和 Alpha
    rgb = img[..., :3]
    alpha = img[..., 3:]

    # 转换 RGB
    v = rgb.astype(np.float32) / 255.0
    mask = v <= 0.0031308
    srgb = np.where(mask, v * 12.92, 1.055 * (v ** (1 / 2.4)) - 0.055)
    srgb = np.clip(srgb, 0.0, 1.0)
    rgb_srgb = (srgb * 255.0 + 0.5).astype(np.uint8)

    # 合并结果
    return np.concatenate([rgb_srgb, alpha], axis=-1)



@njit
def sample_reflection_angle(normal_vec):


    # ---- 3. 生成余弦加权的漫反射方向（理想朗伯表面） ----
    # 在 2D 中，朗伯采样的 PDF = cos(θ)/2，θ 是与法线的夹角
    sin_theta = random.uniform(-1.0, 1.0)        # sinθ 均匀分布
    cos_theta = math.sqrt(1.0 - sin_theta * sin_theta)
    # 切线：法线逆时针旋转90°
    tangent = np.array([-normal_vec[1], normal_vec[0]])
    diffuse_dir = cos_theta * normal_vec + sin_theta * tangent
    diffuse_angle = math.atan2(diffuse_dir[1], diffuse_dir[0])

    # 纯漫反射直接返回

    #if roughness >= 1.0:
        #return diffuse_angle
    return diffuse_angle


@njit
def is_occluded_by_height(start_xy, end_xy, height_map, size):
    """
    使用最初的方法：沿光线投影路径采样，检查光线高度与地形高度的差值是否发生符号翻转。
    如果发生翻转（即从高于地形变成低于地形，或反之），则判定为遮挡。
    平坦区域因为差值始终为 0 或同号，不会被误判。
    """
    x0, y0 = start_xy
    x1, y1 = end_xy

    # 起点和终点的高度（双线性插值）
    h_start = bilinear_interpolate(height_map, x0, y0)
    h_end   = bilinear_interpolate(height_map, x1, y1)

    dx = x1 - x0
    dy = y1 - y0
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-6:
        return False

    # 采样步数：至少每像素一个采样点
    steps = int(math.ceil(dist)) + 1   # 包含两端点

    # 初始差值（在起点处，光线高度等于起点地形高度，差值为0）
    prev_diff = 0.0

    for i in range(1, steps):          # 跳过 i=0（起点）
        t = i / (steps - 1)
        x = x0 + t * dx
        y = y0 + t * dy
        terrain_h = bilinear_interpolate(height_map, x, y)
        ray_h = h_start + (h_end - h_start) * t
        diff = ray_h - terrain_h

        # 符号翻转 → 相交（遮挡）
        if prev_diff * diff < 0:
            return True

        # 注意：我们不在这里用 diff == 0 判断遮挡，
        # 否则平坦区域会全部误判为遮挡。
        # 只有真正的符号翻转才表示光线穿过了地形表面。

        prev_diff = diff

    return False




@njit
def getColorAtSDF(IMG,HEIGHT, SDF, ra, recuDep, Bounce, size,nowcolor):
    """递归获取光线颜色"""
    k = 0
    if recuDep > Bounce:

        return np.array([0, 0, 0, 0], dtype=np.int64)

    #if bli.bilinear_interpolate(SDF, ra.o[0], ra.o[1]) <= 0.0001:
    #pixel = IMG.getpixel((round(ra.o[0]),round(ra.o[1])))
        #print(pixel[3])


    #else:
    #    return (0, 0, 0, 0)
    #print(recuDep)
    #print(recuDep)
    oldk=0





    a=0

    starting_height= HEIGHT[round(ra[0][1])][round(ra[0][0])]

    while True:
        pos = rayp(ra, k)

        ix, iy = round(pos[0]), round(pos[1])
        if ix < 0 or ix >= size[0] or iy < 0 or iy >= size[1]:
            return np.array([0, 0, 0, 0], dtype=np.int64)
            #print(1)
        if bilinear_interpolate(SDF, pos[0], pos[1]) <= 0.00001:


            #return (255,255,255,255)
            pos = rayp(ra, k+0.01)
            ix, iy = round(pos[0]), round(pos[1])
            if ix < 0 or ix >= size[0] or iy < 0 or iy >= size[1]:
                return np.array([0, 0, 0, 0], dtype=np.int64)
            pixel = IMG[iy][ix]

            if a == 0:
                if pixel[3]==255:

                    k += max(abs(bilinear_interpolate(SDF, pos[0], pos[1])),1)
                    if 0<pixel[3]<255:
                        break
                    continue

                #return np.array([0, 0, 0, 255], dtype=np.int64)
            #elif pixel[3]!=0:
                #if a==0:
                #    a=1
                #    continue
            #    if a == 0:
            #        return np.array([pixel[0],pixel[1],pixel[2],255], dtype=np.int64)


            #print(pixel)
            # print(pixel[3])
            #if pixel[3] != 255:
            #    return pixel

            if pixel[3] != 255 and pixel[3]!=0:  # 光源
                # ---- 高度遮挡检测 ----
                if is_occluded_by_height(ra[0], pos, HEIGHT, size):
                    return np.array([0, 0, 0, 0], dtype=np.int64)
                # ---- 原逻辑 ----
                return re(pixel, nowcolor)


            #i = k
            #while bli.bilinear_interpolate(SDF, pos[0], pos[1]) <= 0.0001:
            #    pos = rayp(ra, i)
            #   i -= 0.001
            #n = normal_at(SDF, round(pos[0]), round(pos[1]))
            t_surf, pos_surf = find_surface(ra, oldk, k, SDF)
            if is_occluded_by_height(ra[0], pos_surf, HEIGHT, size):
                return np.array([0, 0, 0, 0], dtype=np.int64)
            #pixel = IMG.getpixel((math.ceil(pos_surf[0]), math.ceil(pos_surf[1])))

            n = normal_at(SDF, pos_surf[0], pos_surf[1])

            if np.dot(ra[1], n) > 0:
                n = -n
            #reflect_dir = ra.d - 2 * np.dot(ra.d, n) * n
            out_angle = sample_reflection_angle(n)
            direction = np.array([math.cos(out_angle), math.sin(out_angle)])
            new_origin = pos_surf + direction * 1e-4  # 或沿法线偏移 n * 1e-4



            #print(bli.bilinear_interpolate(SDF, new_origin[0], new_origin[1]))
            new_ray = (new_origin, direction)

            nc=re(pixel,nowcolor)

            #print(re(pixel,nowcolor))

            if nc[0]<=1:
                if nc[1] < 1:
                    if nc[2] < 1:
                        #print(1)
                        return np.array([0, 0, 0, 0], dtype=np.int64)

            result = getColorAtSDF(IMG, HEIGHT,SDF, new_ray, recuDep + 1, Bounce, size, nc)
            #print(result)

            return result

        else:
            a=1
        oldk = k
        k += abs(bilinear_interpolate(SDF, pos[0], pos[1]))

    return np.array([0, 0, 0, 0], dtype=np.int64)


import numpy as np
from PIL import Image


def replace_non_transparent(image_path_or_array, target_rgba):
    """
    将图片中非透明区域的像素替换为指定的 RGBA 颜色。

    参数:
        image_path_or_array: 图片路径 (str) 或 PIL.Image 对象，或 numpy 数组 (H,W,4)
        target_rgba: tuple (R, G, B, A)，每个分量 0-255，例如 (255,0,0,255) 纯红不透明

    返回:
        PIL.Image 对象 (RGBA 模式)
    """
    # 1. 加载图片并确保为 RGBA 模式
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert("RGBA")
    elif isinstance(image_path_or_array, Image.Image):
        img = image_path_or_array.convert("RGBA")
    else:
        # 假设是 numpy 数组，形状 (H, W, 4)
        img = Image.fromarray(image_path_or_array, "RGBA")

    # 转换为 numpy 数组 (H, W, 4)
    arr = np.array(img, dtype=np.uint8)

    # 2. 提取 alpha 通道 (第 3 通道)
    alpha = arr[:, :, 3]

    # 3. 创建掩码：非透明区域 (alpha > 0)
    mask = alpha > 0

    # 4. 将掩码对应的像素替换为目标 RGBA
    # 注意：target_rgba 需要是 numpy 数组，且形状为 (4,)，并转换为 uint8
    target = np.array(target_rgba, dtype=np.uint8)
    arr[mask] = target

    # 5.（可选）将透明区域的 RGB 也设为 0，避免残留数据
    # 但这一步不是必须的，因为 alpha=0 的像素在显示时完全透明
    arr[~mask] = [0, 0, 0, 0]  # 全透明

    # 6. 转换回 PIL Image 并返回
    result_img = Image.fromarray(arr, "RGBA")
    return result_img


# 使用示例
#colorize_non_transparent_fast("input.png", (255, 0, 0), "output_fast.png")




def invert_by_mask(bg: Image.Image, mask: Image.Image) -> Image.Image:
    """
    根据遮罩图片的透明形状，对背景图片的对应区域进行颜色反色。
    遮罩图片与背景图片左上角对齐，遮罩尺寸必须 ≤ 背景尺寸。

    Args:
        bg: 背景图片（RGB 或 RGBA 模式均可）
        mask: 遮罩图片（建议 RGBA 模式，其 Alpha 通道定义反色区域形状）

    Returns:
        处理后的背景图片（RGB 模式，透明区域若原背景有透明则会被丢弃）

    Raises:
        ValueError: 遮罩图片尺寸大于背景图片时抛出
    """
    # 确保背景为 RGB 模式（避免反色影响 Alpha 通道）
    if bg.mode != 'RGB':
        bg = bg.convert('RGB')

    # 确保遮罩为 RGBA 模式，以便提取 Alpha 通道
    if mask.mode != 'RGBA':
        mask = mask.convert('RGBA')

    # 尺寸校验
    if mask.width > bg.width or mask.height > bg.height:
        raise ValueError("遮罩图片尺寸必须小于等于背景图片尺寸")

    # 提取遮罩图片的 Alpha 通道作为形状遮罩
    alpha_mask = mask.split()[-1]   # 灰度图，尺寸与 mask 相同

    # 遮罩覆盖的区域（左上角对齐）
    box = (0, 0, mask.width, mask.height)

    # 裁切背景中的对应区域
    bg_region = bg.crop(box)

    # 对该区域进行颜色反色
    inverted_region = ImageChops.invert(bg_region)

    # 根据 Alpha 遮罩合成：白色区域显示反色结果，黑色区域保留原背景
    blended_region = Image.composite(inverted_region, bg_region, alpha_mask)

    # 将处理后的区域贴回背景
    bg.paste(blended_region, (0, 0))

    return bg


# ---------- 渲染函数（参数外部传入）----------

def render_loop(img_array, height_array,cimg_array, img_out, sdf, size, samples, bounce):
    current_time2 = 0
    img2_array=img_out
    for i in range(size[0]):
        if i != 0:
            progress = i / size[0] * 100

            remain = (current_time2) * (size[0] - i)
            bar = '#' * round(progress) + ' ' * round(100 - progress)
            print(f'[{bar}] {progress:.1f}% remain time: {remain:.2f}s')

        current_time = time.time()
        for j in range(size[1]):
            nowcolor = np.array([0, 0, 0, 0])
            nowpixel = img_array[j][i]
            b=0
            if nowpixel[3] == 255:
                b=1
            #    img2_array[j][i] = (0, 0, 0, 255)
            #    continue
            elif nowpixel[3] != 0:
                b=2
            #    img2_array[j][i] = img_array[j][i]
            #    img2_array[j][i][3] = 255
            #    continue
            for sp in range(samples):
                # tcol = img.getpixel((i,j))

                sector = 2 * math.pi * (sp + random.random()) / samples
                direction = np.array([math.sin(sector), math.cos(sector)])
                # jd=random.uniform(1,360)
                # jd=jd*math.pi/180
                # direction = np.array([math.sin(jd), math.cos(jd)])
                r1 = (np.array([i + random.random() - 0.5, j + random.random() - 0.5]), direction)
                # col = np.array(getColorAtSDF(img, dst, r1, 0, Bounce, size , (255,255,255,255)))
                col = np.array(getColorAtSDF(img_array,height_array, sdf, r1, 0, bounce, size, np.array([255, 255, 255, 255], dtype=np.int64)))
                # print(col)
                #if b==0:

                #    color = re(cimg_array[j][i], col)
                #else:
                #    color = re(img_array[j][i], col)
                nowcolor = nowcolor + col

            for k in range(len(nowcolor)):
                nowcolor[k] = int(nowcolor[k] / samples)
            if b==0:
                img2_array[j][i] = re(nowcolor, cimg_array[j][i])
            elif b==1:
                img2_array[j][i] = re(nowcolor, img_array[j][i])
            else:
                img2_array[j][i] = nowcolor

        current_time2 = time.time() - current_time
    return img2_array



def render(img_path, height_path,color_path, output_path, Sample, Bounce):
    # 加载输入图像
    img = Image.open(img_path)
    cimg = Image.open(color_path)
    height = Image.open(height_path)

    size = np.array(img.size)
    print(img.info)
    img_array = np.array(img)
    cimg_array = np.array(cimg)
    height_array = np.array(height)
    img_array=srgb_to_linear_rgba(img_array)
    cimg_array = srgb_to_linear_rgba(cimg_array)
    #print(len(img_array[1]))
    img3=replace_non_transparent(img,(0,0,0,255))
    img3bk=invert_by_mask(cimg,img3)
    img3bk=img3bk.crop((0,0,size[0],size[1]))
    img3bk.save(output_path+"_original.png")

    # 生成 alpha 通道掩码（行优先）
    blackAndWhite = [
        [img.getpixel((x, y))[3] > 0 for x in range(size[0])]
        for y in range(size[1])
    ]

    # 计算有符号距离场
    dst = sdf_2d(blackAndWhite)
    dst = np.array(dst)

    # 可选：保存距离场可视化（调试用）
    dist_np = np.array(dst, dtype=np.float32)
    dist_np = (dist_np / dist_np.max()) * 255
    sdf_img = Image.fromarray(dist_np.astype(np.uint8), mode='L')
    sdf_img.save(output_path+'_distance_field.png')

    # 输出画布
    img2 = Image.new("RGBA", (size[0], size[1]), (0, 00, 00))

    img2_array = np.array(img2)






    img2_array=linear_to_srgb_rgba(render_loop(img_array,height_array,cimg_array,img2_array,dst,size,Sample, Bounce))
    # 保存最终结果
    img2=Image.fromarray(img2_array)
    img2.save(output_path)


# ---------- 全局配置（在函数外定义）----------
input_image = "精灵-0001.png"
color_image = "ba.png"
height_image = "height.png"
sample_count = 100
bounce_count = 5
output_file = f"output3/output_{sample_count}_Sample(s)_with_{bounce_count}_Bounce(s)_aa.png"
# 调用渲染函数

#print(linear_to_srgb_rgba(np.array([255,215,255,254])))
rendertime=time.time()

#print(linear_to_srgb_rgba(np.array([255,215,255,254])))

render(input_image,height_image,color_image, output_file, sample_count, bounce_count)
rendertime=time.time()-rendertime
print(rendertime)