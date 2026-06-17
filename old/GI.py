#from numba.core.typing.builtins import Print
#from PIL import paste
from PIL import ImageChops
from old import sdf, ray as r
import bilinearInterpolate as bli
import time
#import numba
import math
import random

random.seed(time.time())



# ---------- 辅助函数 ----------






def rayp(ra, t):
    return ra.o + t * ra.d

def addlist(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def re(a, b):
    r = int(a[0]) * int(b[0]) // 255  # 使用整数除法 // 替代 math.floor
    g = int(a[1]) * int(b[1]) // 255
    b_ = int(a[2]) * int(b[2]) // 255
    return np.array([r, g, b_, 255])

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




def sample_reflection_angle(normal_vec):
    """
    根据粗糙度、法线绝对角度、入射绝对角度，采样一个出射绝对角度。

    参数：
        roughness : float, 0 = 完美镜面, 1 = 完全漫反射（朗伯余弦加权）
        normal_vec : float, 表面法线的世界角度（弧度）
        incident_vec : float, 入射光线的世界角度（弧度）

    返回：
        float : 出射光线的世界角度（弧度）
    """
    # ---- 1. 构造单位向量 ----
    #n = normal_vec#np.array([math.cos(normal_vec), math.sin(normal_vec)])
    #i = incident_vec#np.array([math.cos(incident_vec), math.sin(incident_vec)])
    """
    # ---- 2. 计算完美镜面反射方向 ----
    dot_ni = np.dot(incident_vec, normal_vec)
    reflect = incident_vec - 2 * dot_ni * normal_vec
    specular_angle = math.atan2(reflect[1], reflect[0])

    # 纯镜面直接返回
    if roughness <= 0.0:
        return specular_angle
    """
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

    """
    # ---- 4. 中间粗糙度：概率混合（俄罗斯轮盘赌） ----
    # 以 roughness 为概率选择漫反射，1-roughness 选择镜面
    if random.random() < roughness:
        return diffuse_angle
    else:
        return specular_angle
    """

def getColorAtSDF(IMG, SDF, ra, recuDep, Bounce, size,nowcolor):
    """递归获取光线颜色"""
    k = 0
    if recuDep > Bounce:
        return (0,0,0,0)

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
            #print(1)
        if bli.bilinear_interpolate(SDF, pos[0], pos[1]) <= 0.00001:
            #return (255,255,255,255)
            pos = rayp(ra, k+0.01)
            ix, iy = round(pos[0]), round(pos[1])
            if ix < 0 or ix >= size[0] or iy < 0 or iy >= size[1]:
                return (0, 0, 0, 0)
            pixel = IMG[iy][ix]
            #print(pixel)
            # print(pixel[3])
            #if pixel[3] != 255:
            #    return pixel
            if pixel[3] != 255 and pixel[3]!=0:  # 光源
                #print(pixel)
                #print(1)
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

            #print(re(pixel,nowcolor))

            if nc[0]<=1:
                if nc[1] < 1:
                    if nc[2] < 1:
                        #print(1)
                        return (0,0,0,0)

            result = getColorAtSDF(IMG, SDF, new_ray, recuDep + 1, Bounce, size, nc)
            #print(result)
            return result
        oldk = k
        k += abs(bli.bilinear_interpolate(SDF, pos[0], pos[1]))

    return (0, 0, 0, 0)


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
def render(img_path, color_path, output_path, Sample, Bounce):
    # 加载输入图像
    img = Image.open(img_path)
    cimg = Image.open(color_path)


    size = np.array(img.size)
    print(img.info)
    img_array = np.array(img)
    cimg_array = np.array(cimg)

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
    dst = sdf.sdf_2d(blackAndWhite)
    dst = np.array(dst)

    # 可选：保存距离场可视化（调试用）
    dist_np = np.array(dst, dtype=np.float32)
    dist_np = (dist_np / dist_np.max()) * 255
    sdf_img = Image.fromarray(dist_np.astype(np.uint8), mode='L')
    sdf_img.save('output/distance_field.png')

    # 输出画布
    img2 = Image.new("RGBA", (size[0], size[1]), (0, 00, 00))

    img2_array = np.array(img2)





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
            nowpixel = img_array[j][i]
            if nowpixel[3]==255:
                img2_array[j][i]=(0,0,0,255)
                continue
            elif nowpixel[3]!=0:
                img2_array[j][i]=img_array[j][i]
                img2_array[j][i][3]=255
                continue
            for sp in range(Sample):
                #tcol = img.getpixel((i,j))


                sector = 2 * math.pi * (sp + random.random()) / Sample
                direction = np.array([math.sin(sector), math.cos(sector)])
                #jd=random.uniform(1,360)
                #jd=jd*math.pi/180
                #direction = np.array([math.sin(jd), math.cos(jd)])
                r1 = r.ray(np.array([i+random.random()-0.5, j+random.random()-0.5]), direction)
                #col = np.array(getColorAtSDF(img, dst, r1, 0, Bounce, size , (255,255,255,255)))
                col = np.array(getColorAtSDF(img_array, dst, r1, 0, Bounce, size, (255, 255, 255, 255)))
                #print(col)
                color = re(cimg_array[j][i], col)
                nowcolor = addlist(nowcolor, color)

            for k in range(len(nowcolor)):
                nowcolor[k] = int(nowcolor[k] / Sample)

            img2_array[j][i]=re(nowcolor,cimg_array[j][i])
        current_time2 = time.time() - current_time
    img2_array=linear_to_srgb_rgba(img2_array)
    # 保存最终结果
    img2=Image.fromarray(img2_array)
    img2.save(output_path)


# ---------- 全局配置（在函数外定义）----------
input_image = "精灵-0001.png"
color_image = "background.png"

sample_count = 50
bounce_count = 0
output_file = f"output3/output_{sample_count}_Sample(s)_with_{bounce_count}_Bounce(s)_aa.png"
# 调用渲染函数

#print(linear_to_srgb_rgba(np.array([255,215,255,254])))
rendertime=time.time()

#print(linear_to_srgb_rgba(np.array([255,215,255,254])))

render(input_image, color_image, output_file, sample_count, bounce_count)
rendertime=time.time()-rendertime
print(rendertime)