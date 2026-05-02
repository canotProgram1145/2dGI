import math
import time
import random
import numpy as np
from PIL import Image
import taichi as ti
import sdf
import bilinearInterpolate as bli
import os

# 初始化Taichi (使用GPU加速)
ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)


# ---------- Taichi数据结构定义 ----------
@ti.data_oriented
class Renderer:
    def __init__(self, img_path, color_path, size):
        self.size = size

        # 加载图像数据
        img = Image.open(img_path)
        cimg = Image.open(color_path)

        # 转换为numpy数组
        img_data_np = np.array(img)
        color_data_np = np.array(cimg)

        # 创建Taichi fields
        self.sdf_field = ti.field(ti.f32, shape=(size[1], size[0]))
        self.img_field = ti.field(ti.u8, shape=(size[1], size[0], 4))
        self.color_field = ti.field(ti.u8, shape=(size[1], size[0], 4))
        self.result_field = ti.field(ti.u8, shape=(size[1], size[0], 4))

        # 生成SDF
        blackAndWhite = [
            [img.getpixel((x, y))[3] > 0 for x in range(size[0])]
            for y in range(size[1])
        ]
        dst = sdf.sdf_2d(blackAndWhite)

        # 将SDF数据复制到Taichi field
        for y in range(size[1]):
            for x in range(size[0]):
                self.sdf_field[y, x] = dst[y][x]

        # 将图像数据复制到Taichi fields
        for y in range(size[1]):
            for x in range(size[0]):
                for c in range(4):
                    self.img_field[y, x, c] = img_data_np[y, x, c]
                    self.color_field[y, x, c] = color_data_np[y, x, c]

    @ti.func
    def bilinear_interpolate_ti(self, x: ti.f32, y: ti.f32) -> ti.f32:
        """Taichi版本的双线性插值"""
        x_clamped = ti.min(ti.max(x, 0.0), self.size[0] - 1.0)
        y_clamped = ti.min(ti.max(y, 0.0), self.size[1] - 1.0)

        x0 = ti.floor(x_clamped).cast(ti.i32)
        y0 = ti.floor(y_clamped).cast(ti.i32)
        x1 = x0 + 1
        y1 = y0 + 1

        if x1 >= self.size[0]:
            x1 = self.size[0] - 1
        if y1 >= self.size[1]:
            y1 = self.size[1] - 1

        # 获取四个角点的值
        q00 = self.sdf_field[y0, x0]
        q01 = self.sdf_field[y0, x1]
        q10 = self.sdf_field[y1, x0]
        q11 = self.sdf_field[y1, x1]

        # 计算权重
        dx = x_clamped - x0
        dy = y_clamped - y0

        # 双线性插值
        top = q00 * (1.0 - dx) + q01 * dx
        bottom = q10 * (1.0 - dx) + q11 * dx
        return top * (1.0 - dy) + bottom * dy

    @ti.func
    def normal_at_ti(self, x: ti.f32, y: ti.f32, epsilon: ti.f32 = 1e-4) -> ti.types.vector(2, ti.f32):
        """计算法线向量"""
        dx = self.bilinear_interpolate_ti(x + epsilon, y) - self.bilinear_interpolate_ti(x - epsilon, y)
        dy = self.bilinear_interpolate_ti(x, y + epsilon) - self.bilinear_interpolate_ti(x, y - epsilon)

        grad = ti.Vector([dx, dy])
        norm = grad.norm()

        if norm < 1e-8:
            return ti.Vector([0.0, 0.0])

        return grad / norm

    @ti.func
    def find_surface_ti(self, ray_o: ti.types.vector(2, ti.f32), ray_d: ti.types.vector(2, ti.f32),
                        t_out: ti.f32, t_in: ti.f32, max_iter: ti.i32 = 50, tol: ti.f32 = 1e-6) -> ti.types.vector(3,
                                                                                                                   ti.f32):
        """在Taichi中实现表面查找"""
        t_mid = 0.0
        pos_mid = ti.Vector([0.0, 0.0])

        for _ in range(max_iter):
            t_mid = (t_out + t_in) * 0.5
            pos_mid = ray_o + t_mid * ray_d
            d = self.bilinear_interpolate_ti(pos_mid[0], pos_mid[1])

            if ti.abs(d) < tol or (t_in - t_out) < tol:
                break

            if d > 0:  # 在外部
                t_out = t_mid
            else:  # 在内部
                t_in = t_mid

        return ti.Vector([t_mid, pos_mid[0], pos_mid[1]])

    @ti.func
    def get_color_at_sdf_ti(self, ray_o: ti.types.vector(2, ti.f32), ray_d: ti.types.vector(2, ti.f32),
                            recu_dep: ti.i32, max_bounce: ti.i32) -> ti.types.vector(4, ti.u8):
        """Taichi版本的光线追踪函数"""
        if recu_dep > max_bounce:
            return ti.Vector([0, 0, 0, 0], dt=ti.u8)

        # 检查起点是否在物体内部
        if self.bilinear_interpolate_ti(ray_o[0], ray_o[1]) <= 0.0001:
            ix = ti.min(ti.max(ti.floor(ray_o[0]).cast(ti.i32), 0), self.size[0] - 1)
            iy = ti.min(ti.max(ti.floor(ray_o[1]).cast(ti.i32), 0), self.size[1] - 1)

            pixel = ti.Vector([
                self.img_field[iy, ix, 0],
                self.img_field[iy, ix, 1],
                self.img_field[iy, ix, 2],
                self.img_field[iy, ix, 3]
            ])

            if pixel[3] != 255:
                return pixel.cast(ti.u8)
            return ti.Vector([0, 0, 0, 0], dt=ti.u8)

        k = 0.0
        old_k = 0.0

        while True:
            pos = ray_o + k * ray_d
            ix = ti.floor(pos[0]).cast(ti.i32)
            iy = ti.floor(pos[1]).cast(ti.i32)

            # 检查边界
            if ix < 0 or ix >= self.size[0] or iy < 0 or iy >= self.size[1]:
                return ti.Vector([0, 0, 0, 0], dt=ti.u8)

            d_val = self.bilinear_interpolate_ti(pos[0], pos[1])

            if d_val <= 0.0001:
                # 找到表面
                surface_info = self.find_surface_ti(ray_o, ray_d, old_k, k)
                t_surf = surface_info[0]
                pos_surf = ti.Vector([surface_info[1], surface_info[2]])

                # 获取表面点颜色
                surf_ix = ti.min(ti.max(ti.floor(pos_surf[0]).cast(ti.i32), 0), self.size[0] - 1)
                surf_iy = ti.min(ti.max(ti.floor(pos_surf[1]).cast(ti.i32), 0), self.size[1] - 1)
                pixel = ti.Vector([
                    self.img_field[surf_iy, surf_ix, 0],
                    self.img_field[surf_iy, surf_ix, 1],
                    self.img_field[surf_iy, surf_ix, 2],
                    self.img_field[surf_iy, surf_ix, 3]
                ])

                # 如果是光源，直接返回
                if pixel[3] != 255:
                    return pixel.cast(ti.u8)

                # 计算法线
                n = self.normal_at_ti(pos_surf[0], pos_surf[1])
                if ray_d.dot(n) > 0:
                    n = -n

                # 计算反射方向
                dot_product = ray_d.dot(n)
                reflect_dir = ray_d - 2.0 * dot_product * n
                reflect_dir = reflect_dir.normalized()

                # 递归调用
                if recu_dep + 1 <= max_bounce:
                    return self.get_color_at_sdf_ti(pos_surf, reflect_dir, recu_dep + 1, max_bounce)
                else:
                    return ti.Vector([0, 0, 0, 0], dt=ti.u8)

            old_k = k
            step = ti.max(d_val, 0.001)  # 防止步长为0
            k += step

            # 防止无限循环
            if k > 1000.0:
                return ti.Vector([0, 0, 0, 0], dt=ti.u8)

    @ti.kernel
    def render_kernel(self, sample_count: ti.i32, bounce_count: ti.i32):
        """主渲染kernel，每个像素独立处理"""
        ti.loop_config(block_dim=256)

        for i, j in ti.ndrange(self.size[0], self.size[1]):
            ray_origin = ti.Vector([i + 0.5, j + 0.5])  # 像素中心

            total_color = ti.Vector([0.0, 0.0, 0.0, 0.0])

            for sp in range(sample_count):
                # 生成随机方向
                rand_val = ti.random(ti.f32)
                angle = 2.0 * math.pi * (sp + rand_val) / sample_count
                direction = ti.Vector([ti.sin(angle), ti.cos(angle)])

                # 获取光线颜色
                color_vec = self.get_color_at_sdf_ti(ray_origin, direction, 0, bounce_count)

                # 混合颜色
                if color_vec[3] != 255:
                    base_color = ti.Vector([
                        self.color_field[j, i, 0],
                        self.color_field[j, i, 1],
                        self.color_field[j, i, 2],
                        255
                    ])

                    # 颜色混合：base_color * color_vec / 255
                    mixed_r = base_color[0] * color_vec[0] / 255.0
                    mixed_g = base_color[1] * color_vec[1] / 255.0
                    mixed_b = base_color[2] * color_vec[2] / 255.0

                    total_color[0] += mixed_r
                    total_color[1] += mixed_g
                    total_color[2] += mixed_b
                    total_color[3] += 255.0

            # 计算平均值
            if sample_count > 0:
                total_color /= sample_count

            # 写入结果
            self.result_field[j, i, 0] = ti.cast(total_color[0], ti.u8)
            self.result_field[j, i, 1] = ti.cast(total_color[1], ti.u8)
            self.result_field[j, i, 2] = ti.cast(total_color[2], ti.u8)
            self.result_field[j, i, 3] = 255  # 确保alpha通道为255

    def render(self, sample_count, bounce_count, output_path):
        """外部调用的渲染函数"""
        start_time = time.time()

        print("开始渲染...")
        # 执行渲染kernel
        self.render_kernel(sample_count, bounce_count)

        # 将结果转换回numpy数组
        result_array = self.result_field.to_numpy()

        # 创建PIL图像
        result_img = Image.fromarray(result_array, mode='RGBA')

        # 保存结果
        result_img.save(output_path)

        total_time = time.time() - start_time
        print(f"渲染完成！总时间: {total_time:.2f}秒")
        print(f"渲染速度: {self.size[0] * self.size[1] / total_time:.2f} 像素/秒")

        return result_img


# ---------- 主程序 ----------
def render_with_taichi(img_path, color_path, output_path, Sample, Bounce):
    # 加载输入图像获取尺寸
    img = Image.open(img_path)
    size = np.array(img.size)

    print(f"开始渲染，分辨率: {size[0]}x{size[1]}")
    print(f"采样数: {Sample}, 反弹次数: {Bounce}")

    # 创建渲染器
    renderer = Renderer(img_path, color_path, size)

    # 执行渲染
    result_img = renderer.render(Sample, Bounce, output_path)

    return result_img


# ---------- 配置和执行 ----------
if __name__ == "__main__":
    input_image = "精灵-0001.png"
    color_image = "img.png"

    sample_count = 200
    bounce_count = 3
    output_file = f"output/output_taichi_{sample_count}S_{bounce_count}B.png"

    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)

    # 调用Taichi渲染函数
    render_with_taichi(input_image, color_image, output_file, sample_count, bounce_count)