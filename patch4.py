#!/home/21/xiaohui/anaconda3/envs/machine/lib/python3.9
import openslide
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
import os
from multiprocessing import Pool

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def Count_pixel(img):
    sum = 0
    im = 'img'
    im1 = Counter(eval(im).flatten()).most_common(3)
    for i in range(3):
        if i >= len(im1):
            break
        sum += im1[i][0]
    sum = sum / len(im1)
    return sum


def otsu_thresh(img_gray):
    pixel_sum = img_gray.shape[0] * img_gray.shape[1]
    his, bins = np.histogram(img_gray, np.arange(0, 257))
    final_thresh = -1
    max_value = -1
    intensity = np.arange(256)
    num_organization = np.zeros_like(img_gray)
    num_all = np.ones_like(img_gray)
    for i in bins[1:-1]:
        pc0 = np.sum(his[:i]) + 1
        pc1 = np.sum(his[i:]) + 1
        w0 = pc0 / pixel_sum
        w1 = pc1 / pixel_sum

        u0 = np.sum(intensity[:i] * his[:i]) / float(pc0)
        u1 = np.sum(intensity[i:] * his[i:]) / float(pc1)

        value = w0 * w1 * (u0 - u1) ** 2

        if value > max_value:
            final_thresh = i
            max_value = value
    if final_thresh < 210:
        final_thresh = 210
    num_organization[img_gray < final_thresh] = 1
    organization_area = np.sum(num_organization) / (256*256)
    return organization_area

def get_file_name(from_name):
    names = os.listdir(from_name)
    file_name = []
    for name in names:
        index = name.rfind('.')
        name = name[:index]
        file_name.append(name)
    return file_name

def img_split(file):
    from_name = '/data15/data15_5/zihan/MSIslide/MSS/中肿MSS/'
    to_name = '/data15/data15_5/xiaohui/MSI_patch/MIS/mis_16x/'
    print(file + '.svs')
    slide = openslide.open_slide(from_name + file + '.svs')
    data_gen = DeepZoomGenerator(slide, tile_size=240, overlap=8, limit_bounds=False)
    #[w,h] = data_gen.level_dimensions[-2]
    num_w = data_gen.level_tiles[16][0]
    num_h = data_gen.level_tiles[16][1]
    for i in range(num_w):
        for j in range(num_h):
            ele1 = data_gen.get_tile(16, (i, j)).size[0]
            ele2 = data_gen.get_tile(16, (i, j)).size[1]
            if ele1 != 256 or ele2 != 256:
                continue
            img = np.array(data_gen.get_tile(16, (i, j)))
            img_gray = Image.fromarray(img.astype('uint8')).convert('L')
            img_gray = np.array(img_gray)
            area = otsu_thresh(img_gray)
            sum = Count_pixel(img)
            if area > 0.3 and sum < 240:
                plt.imsave((to_name + file + '/' + file + '_' + str(i) + '_' + str(j) + '.jpg'), img)
    print(file + ' finish')


if __name__ == "__main__":
    from_name = '/data15/data15_5/zihan/MSIslide/MSS/中肿MSS/'
    to_name = '/data15/data15_5/xiaohui/MSI_patch/MIS/mis_16x/'
    file_name = get_file_name(from_name)
    p = Pool(processes=10)
    p.map(img_split, file_name)
    p.close()
    p.join()