import math
import time
import random
import numpy as np
from PIL import Image
#from numba.core.typing.builtins import Print


import ray as r
import sdf
import bilinearInterpolate as bli
import time
#import numba

random.seed(time.time())


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

import math
import random
import numpy as np

def sample_reflection_angle(roughness, normal_angle, incident_angle):
    """
    根据粗糙度、法线绝对角度、入射绝对角度，采样一个出射绝对角度。

    参数：
        roughness : float, 0 = 完美镜面, 1 = 完全漫反射（朗伯余弦加权）
        normal_angle : float, 表面法线的世界角度（弧度）
        incident_angle : float, 入射光线的世界角度（弧度）

    返回：
        float : 出射光线的世界角度（弧度）
    """
    # ---- 1. 构造单位向量 ----
    #n = normal_angle#np.array([math.cos(normal_angle), math.sin(normal_angle)])
    #i = incident_angle#np.array([math.cos(incident_angle), math.sin(incident_angle)])

    # ---- 2. 计算完美镜面反射方向 ----
    dot_ni = np.dot(incident_angle, normal_angle)
    reflect = incident_angle - 2 * dot_ni * normal_angle
    specular_angle = math.atan2(reflect[1], reflect[0])

    # 纯镜面直接返回
    if roughness <= 0.0:
        return specular_angle

    # ---- 3. 生成余弦加权的漫反射方向（理想朗伯表面） ----
    # 在 2D 中，朗伯采样的 PDF = cos(θ)/2，θ 是与法线的夹角
    sin_theta = random.uniform(-1.0, 1.0)        # sinθ 均匀分布
    cos_theta = math.sqrt(1.0 - sin_theta * sin_theta)
    # 切线：法线逆时针旋转90°
    tangent = np.array([-normal_angle[1], normal_angle[0]])
    diffuse_dir = cos_theta * normal_angle + sin_theta * tangent
    diffuse_angle = math.atan2(diffuse_dir[1], diffuse_dir[0])

    # 纯漫反射直接返回
    if roughness >= 1.0:
        return diffuse_angle

    # ---- 4. 中间粗糙度：概率混合（俄罗斯轮盘赌） ----
    # 以 roughness 为概率选择漫反射，1-roughness 选择镜面
    if random.random() < roughness:
        return diffuse_angle
    else:
        return specular_angle


def getColorAtSDF(IMG, SDF, ra, recuDep, Bounce, size,nowcolor):
    """递归获取光线颜色"""
    k = 0
    if recuDep > Bounce:
        return (0, 0, 0, 0)

    #if bli.bilinear_interpolate(SDF, ra.o[0], ra.o[1]) <= 0.0001:
    #pixel = IMG.getpixel((round(ra.o[0]),round(ra.o[1])))
        #print(pixel[3])


    #else:
    #    return (0, 0, 0, 0)
    #print(recuDep)
    #print(recuDep)
    oldk=0
    while True:
        pos = rayp(ra, k)
        ix, iy = round(pos[0]), round(pos[1])
        if ix < 0 or ix >= size[0] or iy < 0 or iy >= size[1]:
            return (0, 0, 0, 0)

        if bli.bilinear_interpolate(SDF, pos[0], pos[1]) <= 0.00001:
            #return (255,255,255,255)
            pos = rayp(ra, k+0.01)
            ix, iy = round(pos[0]), round(pos[1])
            if ix < 0 or ix >= size[0] or iy < 0 or iy >= size[1]:
                return (0, 0, 0, 0)
            pixel = IMG.getpixel((ix, iy))

            #print(pixel)
            #if pixel[3] != 255:
            #    return pixel
            if pixel[3] != 255 and pixel[3]!=0:  # 光源
                #print(pixel)
                return re(pixel,nowcolor)

            #i = k
            #while bli.bilinear_interpolate(SDF, pos[0], pos[1]) <= 0.0001:
            #    pos = rayp(ra, i)
            #   i -= 0.001
            #n = normal_at(SDF, round(pos[0]), round(pos[1]))
            t_surf, pos_surf = find_surface(ra, oldk, k, SDF)
            #pixel = IMG.getpixel((math.ceil(pos_surf[0]), math.ceil(pos_surf[1])))

            n = normal_at(SDF, pos_surf[0], pos_surf[1])

            if np.dot(ra.d, n) > 0:
                n = -n
            #reflect_dir = ra.d - 2 * np.dot(ra.d, n) * n
            out_angle = sample_reflection_angle(1, n, ra.d)
            direction = np.array([math.cos(out_angle), math.sin(out_angle)])
            new_origin = pos_surf + direction * 1e-4  # 或沿法线偏移 n * 1e-4
            #print(bli.bilinear_interpolate(SDF, new_origin[0], new_origin[1]))
            new_ray = r.ray(new_origin, direction)

            nc=re(pixel,nowcolor)
            #print(nc)

            if nc[0]<=1:
                if nc[1] < 1:
                    if nc[2] < 1:
                        return (0,0,0,0)
            result = getColorAtSDF(IMG, SDF, new_ray, recuDep + 1, Bounce, size, nc)
            return result
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

            remain = (current_time2) * (size[0] - i)
            bar = '#' * round(progress) + ' ' * round(100-progress)
            print(f'[{bar}] {progress:.1f}% remain time: {remain:.2f}s')

        current_time = time.time()
        for j in range(size[1]):
            nowcolor = [0, 0, 0, 0]

            if img.getpixel((i, j))[3]==255:
                img2.putpixel((i,j),(0,0,0,255))
                continue
            elif img.getpixel((i,j))[3]!=0:
                img2.putpixel((i,j),img.getpixel((i,j)))
                continue
            for sp in range(Sample):
                tcol = img.getpixel((i,j))


                sector = 2 * math.pi * (sp + random.random()) / Sample
                direction = np.array([math.sin(sector), math.cos(sector)])
                #jd=random.uniform(1,360)
                #jd=jd*math.pi/180
                #direction = np.array([math.sin(jd), math.cos(jd)])
                r1 = r.ray(np.array([i+random.random()-0.5, j+random.random()-0.5]), direction)
                col = np.array(getColorAtSDF(img, dst, r1, 0, Bounce, size , (255,255,255,255)))

                color = re(cimg.getpixel((i, j)), col)
                nowcolor = addlist(nowcolor, color)

            for k in range(len(nowcolor)):
                nowcolor[k] = int(nowcolor[k] / Sample)
            img2.putpixel((i, j), tuple(nowcolor))
        current_time2 = time.time() - current_time

    # 保存最终结果
    img2.save(output_path)

# ---------- 全局配置（在函数外定义）----------
input_image = "精灵-0001.png"
color_image = "ba.png"

sample_count = 1

bounce_count = 1
output_file = f"output/output_{sample_count}_Sample(s)_with_{bounce_count}_Bounce(s)_without_linear.png"
# 调用渲染函数
rendertime=time.time()

render(input_image, color_image, output_file, sample_count, bounce_count)
rendertime=time.time()-rendertime
print(rendertime)