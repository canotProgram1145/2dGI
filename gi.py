import math
import time
import random
import numpy as np
from PIL import Image
import ray as r
import sdf
import bilinearInterpolate as bli

# ---------- 辅助函数 ----------
def rayp(ra, t):
    return ra.o + t * ra.d

def addlist(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def re(a, b):
    return np.array([int(a[0] * b[0] / 255),
                     int(a[1] * b[1] / 255),
                     int(a[2] * b[2] / 255), 255])

def normal_at(sdf, x, y, epsilon=1e-4):
    """返回 (x,y) 处的单位法线（指向外部）"""
    dx = (bli.bilinear_interpolate(sdf, x + epsilon, y) -
          bli.bilinear_interpolate(sdf, x - epsilon, y))
    dy = (bli.bilinear_interpolate(sdf, x, y + epsilon) -
          bli.bilinear_interpolate(sdf, x, y - epsilon))
    grad = np.array([dx, dy])
    norm = np.linalg.norm(grad)
    if norm == 0:
        return np.array([0, 0])
    return grad / norm

def find_surface(ra, t_out, t_in, SDF, max_iter=100, tol=1e-7):
    """在 t_out (外部) 和 t_in (内部) 之间二分查找表面点"""
    for _ in range(max_iter):
        t_mid = (t_out + t_in) * 0.5
        pos_mid = rayp(ra, t_mid)
        d = bli.bilinear_interpolate(SDF, pos_mid[0], pos_mid[1])
        if abs(d) < tol or (t_in - t_out) < tol:
            return t_mid, pos_mid
        if d > 0:  # 在外部
            t_out = t_mid
        else:      # 在内部
            t_in = t_mid
    # 返回最后的中点
    t_mid = (t_out + t_in) * 0.5
    return t_mid, rayp(ra, t_mid)

def getColorAtSDF(IMG, SDF, ra, recuDep, Bounce, size):
    """递归获取光线颜色"""
    k = 0
    if recuDep > Bounce:
        return (0, 0, 0, 0)

    if bli.bilinear_interpolate(SDF, ra.o[0], ra.o[1]) <= 0.0001:
        pixel = IMG.getpixel(ra.o)
        if pixel[3] != 255:
            return pixel
        else:
            return (0, 0, 0, 0)
    oldk=0
    while True:
        pos = rayp(ra, k)
        ix, iy = round(pos[0]), round(pos[1])
        if ix < 0 or ix >= size[0] or iy < 0 or iy >= size[1]:
            return (0, 0, 0, 0)

        if bli.bilinear_interpolate(SDF, pos[0], pos[1]) <= 0.0001:
            #return (255,255,255,255)
            pixel = IMG.getpixel((ix, iy))
            #if pixel[3] != 255:
            #    return pixel
            if pixel[3] != 255:  # 光源
                return pixel
            #i = k
            #while bli.bilinear_interpolate(SDF, pos[0], pos[1]) <= 0.0001:
            #    pos = rayp(ra, i)
            #    i -= 0.001
            #n = normal_at(SDF, round(pos[0]), round(pos[1]))
            t_surf, pos_surf = find_surface(ra, oldk, k, SDF)
            #pixel = IMG.getpixel((math.ceil(pos_surf[0]), math.ceil(pos_surf[1])))

            n = normal_at(SDF, pos_surf[0], pos_surf[1])
            #print(bli.bilinear_interpolate(SDF, pos_surf[0], pos_surf[1]))
            if np.dot(ra.d, n) > 0:
                n = -n
            reflect_dir = ra.d - 2 * np.dot(ra.d, n) * n
            new_ray = r.ray(pos, reflect_dir)
            return getColorAtSDF(IMG, SDF, new_ray, recuDep + 1, Bounce, size)
        oldk = k
        k += max(bli.bilinear_interpolate(SDF, pos[0], pos[1]),0)

    return (0, 0, 0, 0)

# ---------- 渲染函数（参数外部传入）----------
def render(img_path, color_path, output_path, Sample, Bounce):
    # 加载输入图像
    img = Image.open(img_path)
    cimg = Image.open(color_path)
    size = np.array(img.size)

    # 生成 alpha 通道掩码（行优先）
    blackAndWhite = [
        [img.getpixel((x, y))[3] > 0 for x in range(size[0])]
        for y in range(size[1])
    ]

    # 计算有符号距离场
    dst = sdf.sdf_2d(blackAndWhite)
    dst = np.array(dst)

    # 可选：保存距离场可视化（调试用）
    dist_np = np.array(dst, dtype=np.float32)
    dist_np = (dist_np / dist_np.max()) * 255
    sdf_img = Image.fromarray(dist_np.astype(np.uint8), mode='L')
    sdf_img.save('output/distance_field.png')

    # 输出画布
    img2 = Image.new("RGBA", (size[0], size[1]), (9, 10, 20))

    current_time2 = 0
    for i in range(size[0]):
        if i != 0:
            progress = i / size[0] * 100

            remain = (current_time2 / i) * (size[0] - i)
            bar = '#' * round(progress) + ' ' * round(100-progress)
            print(f'[{bar}] {progress:.1f}% remain time: {remain:.2f}s')

        current_time = time.time()
        for j in range(size[1]):
            nowcolor = [0, 0, 0, 0]
            for sp in range(Sample):
                #sector = 2 * math.pi * (sp + random.random()) / Sample
                #direction = np.array([math.sin(sector), math.cos(sector)])
                jd=random.uniform(1,360)
                jd=jd*math.pi/180
                direction = np.array([math.sin(jd), math.cos(jd)])
                r1 = r.ray(np.array([i, j]), direction)
                col = np.array(getColorAtSDF(img, dst, r1, 0, Bounce, size))
                if col[3] != 255:
                    color = re(cimg.getpixel((i, j)), col)
                    nowcolor = addlist(nowcolor, color)
            for k in range(len(nowcolor)):
                nowcolor[k] = int(nowcolor[k] / Sample)
            img2.putpixel((i, j), tuple(nowcolor))
        current_time2 += time.time() - current_time

    # 保存最终结果
    img2.save(output_path)

# ---------- 全局配置（在函数外定义）----------
input_image = "精灵-0001.png"
color_image = "img.png"

sample_count = 1

bounce_count = 5
output_file = f"output/output_{sample_count}_Sample(s)_with_{bounce_count}_Bounce(s).png"
# 调用渲染函数
render(input_image, color_image, output_file, sample_count, bounce_count)