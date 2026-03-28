from PIL import Image
import math
import numpy as np
import ray as r
import random
import sdf
import bilinearInterpolate as bli


def rayp(ra,t):
    return ra.o+t*ra.d

def addlist(a,b):
    result = []
    for i in range(len(a)):
        result.append(a[i] + b[i])
    return result

def re(a,b):
    return list([a[0] * b[0] / 255, a[1] * b[1] / 255, a[2] * b[2] / 255])









img = Image.open("精灵-0001.png")
size=np.array(img.size)
img2= Image.new("RGB",(size[0],size[1]),(9,10,20))
jd = 90
jd = jd * math.pi / 180
pos=np.array([0,0])
r1=r.ray(np.array([0,0]),np.array([math.sin(jd),math.cos(jd)]))
Sample=10

blackAndWhite=[]
print(size[0])
for i in range(size[0]):
    #print(i)
    blackAndWhite.append([])
    for j in range(size[1]):
        #print(j)
        blackAndWhite[i].append(img.getpixel((i,j))==(120,120,120) or img.getpixel((i,j))==(128,128,249) or img.getpixel((i,j))==(255,255,255))
print(blackAndWhite)
#bool_array = np.array(blackAndWhite, dtype=bool)
#img_array = bool_array.astype(np.uint8) * 255
#image = Image.fromarray(img_array, mode='L')
#image.save('black_white.png')
dst=sdf.sdf_2d(blackAndWhite)
dst=np.array(dst)

#
dist_np = np.array(dst, dtype=np.float32)
dist_np = (dist_np / dist_np.max()) * 255
sdf1 = Image.fromarray(dist_np.astype(np.uint8), mode='L')
sdf1.save('output/distance_field.png')
#print(size[0])
# 将数组转换为图像对象
#image = Image.fromarray(dst, mode='L')  # 'L' 表示灰度图

# 保存图像
#image.save('output.png')


for i in range(size[0]):
    #print(i)
    print(i)
    for j in range(size[1]):
        #k=0
        nowcolor = [0, 0, 0]
        #print(img.getpixel((i,j)))
        for sp in range(Sample):
            #print(pos[0])
            jd=random.randint(1, 360)
            jd=jd*math.pi/180
            r1.o=np.array([i,j])
            r1.d=np.array([math.sin(jd),math.cos(jd)])
            k=0

            while True:
                #k=dst[i][j]


                #pos = np.array([int(pos[0]),int(pos[1])])

                pos = rayp(r1, k)
                #print(pos)
                if pos[0] < 0 or pos[0] >= size[0] or pos[1] < 0 or pos[1] >= size[1]:

                    #nowcolor = addlist(nowcolor, img.getpixel((i,j)))


                    break

                if bli.bilinear_interpolate(dst,pos[1],pos[0]) <= 0.1:

                    if img.getpixel(np.round(pos))==(128,128,249):
                        color=re(img.getpixel((i,j)),[128,128,249])
                        nowcolor=addlist(nowcolor,color)
                        break
                    if img.getpixel(np.round(pos))==(255,255,255):
                        color = re(img.getpixel((i, j)), [255,255,255])
                        nowcolor=addlist(nowcolor,color)
                        #print(color)
                        break
                    if img.getpixel(np.round(pos))==(120,120,120):
                        #color = re(img.getpixel((i, j)), [255, 255, 255])
                        #nowcolor = addlist(nowcolor, color)
                        #print(1)
                        break
                    break


                k += bli.bilinear_interpolate(dst,pos[1],pos[0])
                #print(bli.bilinear_interpolate(dst,pos[1],pos[0]))
                #if dst[int(pos[1])][int(pos[0])]-1<=0:
                    #break

                #print(dst[int(pos[1])][int(pos[0])])
                #if dst[int(pos[1])][int(pos[0])]==0:
                #    break

        for k in range(len(nowcolor)):
            nowcolor[k]=int(nowcolor[k]/Sample)

        img2.putpixel((i,j),tuple(nowcolor))

img2.save(f"output/output {Sample} Sample.png")
#nowcolor=addlist(nowcolor,[128,128,249,255])
