import math
import time
import numpy as np
from PIL import Image
import taichi as ti
import sdf

# ---------- Taichi 初始化 ----------
ti.init(arch=ti.gpu)           # 没有 GPU 请改为 ti.cpu

# ---------- 双线性插值（SDF） ----------
@ti.func
def bilinear_interpolate(
    sdf: ti.template(), x: ti.f32, y: ti.f32, W: ti.i32, H: ti.i32
) -> ti.f32:
    x_clamp = ti.max(0.0, ti.min(x, ti.cast(W - 1, ti.f32) - 1e-4))
    y_clamp = ti.max(0.0, ti.min(y, ti.cast(H - 1, ti.f32) - 1e-4))
    x0 = ti.cast(ti.floor(x_clamp), ti.i32)
    y0 = ti.cast(ti.floor(y_clamp), ti.i32)
    x1 = ti.min(x0 + 1, W - 1)
    y1 = ti.min(y0 + 1, H - 1)
    fx = x_clamp - ti.cast(x0, ti.f32)
    fy = y_clamp - ti.cast(y0, ti.f32)
    v00 = sdf[x0, y0]
    v10 = sdf[x1, y0]
    v01 = sdf[x0, y1]
    v11 = sdf[x1, y1]
    return (
        v00 * (1.0 - fx) * (1.0 - fy)
        + v10 * fx * (1.0 - fy)
        + v01 * (1.0 - fx) * fy
        + v11 * fx * fy
    )

# ---------- 法线计算 ----------
@ti.func
def normal_at(
    sdf: ti.template(), x: ti.f32, y: ti.f32, W: ti.i32, H: ti.i32, eps: ti.f32
) -> ti.Vector:
    dx = bilinear_interpolate(sdf, x + eps, y, W, H) - bilinear_interpolate(
        sdf, x - eps, y, W, H
    )
    dy = bilinear_interpolate(sdf, x, y + eps, W, H) - bilinear_interpolate(
        sdf, x, y - eps, W, H
    )
    grad = ti.Vector([dx, dy])
    norm = grad.norm()
    n = ti.Vector([0.0, 0.0])
    if norm > 1e-8:
        n = grad / norm
    return n

# ---------- 二分查找表面（无内部 return） ----------
@ti.func
def find_surface(
    o_x: ti.f32, o_y: ti.f32, d_x: ti.f32, d_y: ti.f32,
    t_out: ti.f32, t_in: ti.f32,
    sdf: ti.template(), W: ti.i32, H: ti.i32,
) -> ti.Vector:  # 返回 (t, x, y)
    t_res = 0.0
    x_res = 0.0
    y_res = 0.0
    converged = False
    for _ in range(50):
        t_mid = (t_out + t_in) * 0.5
        pos_x = o_x + t_mid * d_x
        pos_y = o_y + t_mid * d_y
        d = bilinear_interpolate(sdf, pos_x, pos_y, W, H)
        if ti.abs(d) < 1e-6 or (t_in - t_out) < 1e-6:
            t_res = t_mid
            x_res = pos_x
            y_res = pos_y
            converged = True
            break
        if d > 0.0:
            t_out = t_mid
        else:
            t_in = t_mid
    if not converged:
        t_res = (t_out + t_in) * 0.5
        x_res = o_x + t_res * d_x
        y_res = o_y + t_res * d_y
    return ti.Vector([t_res, x_res, y_res])

# ---------- 光线追踪 ----------
@ti.func
def get_color(
    o_x: ti.f32, o_y: ti.f32, d_x: ti.f32, d_y: ti.f32,
    max_bounce: ti.i32,
    W: ti.i32, H: ti.i32,
    sdf: ti.template(), img: ti.template(), cimg: ti.template(),
) -> ti.Vector:
    result = ti.Vector([0.0, 0.0, 0.0, 0.0])
    done = False

    cur_o_x = o_x
    cur_o_y = o_y
    cur_d_x = d_x
    cur_d_y = d_y
    depth = 0

    while depth <= max_bounce and not done:
        k = 0.0
        oldk = 0.0
        step_count = 0
        max_steps = 2048

        # 检查起点是否在形状内部（与原版 ra.o 检测一致）
        start_sdf = bilinear_interpolate(sdf, cur_o_x, cur_o_y, W, H)
        if start_sdf <= 0.0001:
            ix = ti.cast(ti.floor(cur_o_x + 0.5), ti.i32)
            iy = ti.cast(ti.floor(cur_o_y + 0.5), ti.i32)
            if 0 <= ix < W and 0 <= iy < H:
                r = img[ix, iy, 0]
                g = img[ix, iy, 1]
                b = img[ix, iy, 2]
                a = img[ix, iy, 3]
                if a != 255:                     # 光源
                    result = ti.Vector([ti.cast(r, ti.f32), ti.cast(g, ti.f32),
                                        ti.cast(b, ti.f32), ti.cast(a, ti.f32)])
                    done = True
                else:
                    done = True                  # 内部非光源 -> 黑色
            else:
                done = True
            break

        # 射线步进
        hit = False
        while step_count < max_steps and not done:
            pos_x = cur_o_x + k * cur_d_x
            pos_y = cur_o_y + k * cur_d_y
            ix = ti.cast(ti.floor(pos_x + 0.5), ti.i32)
            iy = ti.cast(ti.floor(pos_y + 0.5), ti.i32)

            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                done = True
                break

            d_sdf = bilinear_interpolate(sdf, pos_x, pos_y, W, H)
            if d_sdf <= 0.0001:
                # 命中表面，先用整数坐标判断光源
                r = img[ix, iy, 0]
                g = img[ix, iy, 1]
                b = img[ix, iy, 2]
                a = img[ix, iy, 3]
                if a != 255:   # 光源
                    result = ti.Vector([ti.cast(r, ti.f32), ti.cast(g, ti.f32),
                                        ti.cast(b, ti.f32), ti.cast(a, ti.f32)])
                    done = True
                    break

                # 非光源 -> 反射
                surf = find_surface(cur_o_x, cur_o_y, cur_d_x, cur_d_y,
                                    oldk, k, sdf, W, H)
                surf_x = surf[1]
                surf_y = surf[2]
                n = normal_at(sdf, surf_x, surf_y, W, H, 1e-4)
                dot = cur_d_x * n[0] + cur_d_y * n[1]
                if dot > 0.0:
                    n = -n
                reflect_x = cur_d_x - 2.0 * dot * n[0]
                reflect_y = cur_d_y - 2.0 * dot * n[1]
                # 偏移防止自交
                cur_o_x = surf_x + reflect_x * 1e-3
                cur_o_y = surf_y + reflect_y * 1e-3
                cur_d_x = reflect_x
                cur_d_y = reflect_y
                depth += 1
                hit = True
                break
            oldk = k
            k += ti.max(d_sdf, 0.0)
            step_count += 1

        if not hit and not done:
            done = True

    return result

# ---------- 渲染核心 Kernel ----------
@ti.kernel
def render_kernel(
    W: ti.i32, H: ti.i32,
    Sample: ti.i32,
    Bounce: ti.i32,
    sdf_field: ti.template(),
    img_field: ti.template(),
    cimg_field: ti.template(),
    output_field: ti.template(),
):
    for i, j in ti.ndrange(W, H):
        now_color = ti.Vector([0.0, 0.0, 0.0, 0.0])
        # 分层采样：均匀划分扇区
        for sp in range(Sample):
            # sector_base + random_offset
            base_angle = 2.0 * math.pi * ti.cast(sp, ti.f32) / ti.cast(Sample, ti.f32)
            random_offset = ti.random() * 2.0 * math.pi / ti.cast(Sample, ti.f32)
            angle = base_angle + random_offset
            dir_x = ti.sin(angle)
            dir_y = ti.cos(angle)

            col = get_color(
                ti.cast(i, ti.f32),           # 光线起点为像素整数坐标
                ti.cast(j, ti.f32),
                dir_x, dir_y,
                Bounce, W, H,
                sdf_field, img_field, cimg_field,
            )

            base_r = ti.cast(cimg_field[i, j, 0], ti.f32)
            base_g = ti.cast(cimg_field[i, j, 1], ti.f32)
            base_b = ti.cast(cimg_field[i, j, 2], ti.f32)

            now_color[0] += base_r * col[0] / 255.0
            now_color[1] += base_g * col[1] / 255.0
            now_color[2] += base_b * col[2] / 255.0
            now_color[3] += 255.0

        for c in ti.static(range(3)):
            output_field[i, j, c] = ti.cast(
                ti.floor(now_color[c] / Sample + 0.5), ti.u8
            )
        output_field[i, j, 3] = ti.cast(255, ti.u8)

# ---------- Python 侧渲染函数 ----------
def render(img_path, color_path, output_path, Sample, Bounce):
    img = Image.open(img_path).convert("RGBA")
    cimg = Image.open(color_path).convert("RGBA")
    W, H = img.size

    # 生成 SDF（仍用 Python 模块）
    blackAndWhite = [
        [img.getpixel((x, y))[3] > 0 for x in range(W)] for y in range(H)
    ]
    dst = sdf.sdf_2d(blackAndWhite)

    # 转换为 Taichi field（注意转置为 [x, y] 顺序）
    sdf_np = np.array(dst, dtype=np.float32).T
    ti_sdf = ti.field(ti.f32, shape=(W, H))
    ti_sdf.from_numpy(sdf_np)

    img_np = np.array(img, dtype=np.uint8).transpose(1, 0, 2)  # (W, H, 4)
    ti_img = ti.field(ti.u8, shape=(W, H, 4))
    ti_img.from_numpy(img_np)

    cimg_np = np.array(cimg, dtype=np.uint8).transpose(1, 0, 2)
    ti_cimg = ti.field(ti.u8, shape=(W, H, 4))
    ti_cimg.from_numpy(cimg_np)

    ti_output = ti.field(ti.u8, shape=(W, H, 4))

    print("Rendering with Taichi...")
    t0 = time.time()
    render_kernel(W, H, Sample, Bounce, ti_sdf, ti_img, ti_cimg, ti_output)
    ti.sync()
    t1 = time.time()
    print(f"Done in {t1 - t0:.2f}s")

    out_np = ti_output.to_numpy().transpose(1, 0, 2)
    Image.fromarray(out_np, "RGBA").save(output_path)

# ---------- 入口 ----------
if __name__ == "__main__":
    img_path = "精灵-0001.png"
    color_path = "img.png"
    sample_count = 3000
    bounce_count = 30
    output_file = f"output_{sample_count}spp_{bounce_count}bounce.png"
    render(img_path, color_path, output_file, sample_count, bounce_count)