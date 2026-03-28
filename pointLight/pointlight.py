from PIL import Image
import math

size = 1024
img = Image.new("RGB", (size, size), 0)
l = 50  # 衰减尺度参数

for i in range(size):
    for j in range(size):
        dx = i - size/2
        dy = j - size/2
        d = math.sqrt(dx*dx + dy*dy)
        gray = int(255 * l*l / (d*d + l*l))
        img.putpixel((i, j), (gray, gray, gray))

img.save('lightmap_fixed1.png')