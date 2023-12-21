import os
import numpy as np
import cv2 as cv
from tqdm import tqdm


def cut(i, dir, data_list, output_dir):  # 当前图片路径
    filename = data_list[i]
    img = cv.imread(dir, 0)
    edges = cv.Canny(img, 100, 200)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    edge_filename = os.path.join(output_dir, filename)
    print(edge_filename)
    cv.imwrite(edge_filename, edges)
    return edges


def dilate_edges(edges):
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv.dilate(edges, kernel, iterations=1)
    return dilated_edges

def read_images(root, mode):  # 读取文件list名字
    # 可以考虑的 mode 只有 ['train', 'val']
    
    label_dir = os.path.join(root, mode, 'label')


    data_list3 = os.listdir(label_dir)
    labels = []

    print('%s image loading' % mode)
    

    for it in tqdm(data_list3):
        # print(it)
        if (it[-4:] == '.jpg' or it[-4:]=='.png'):  # 只读取png或jpg文件
            label_path = os.path.join(label_dir, it)
            labels.append(label_path)
     
        
    return data_list3, labels

def process_images(root, mode, output_dirs):  # 读取文件list名字
    # 可以考虑的 mode 只有 ['train', 'val']
    label_dir = os.path.join(root, mode, 'label')
    
    
    data_list3 = os.listdir(label_dir)
    labels = []

    print('%s image loading' % mode)
   
    for it in tqdm(data_list3):
        # print(it)
        if (it[-4:] == '.jpg' or it[-4:] == '.png'):  # 只读取png或jpg文件
            label_path = os.path.join(label_dir, it)
            labels.append(label_path)

    for i, element in enumerate(labels):
        print(i, element)
        edges = cut(i, element, data_list, os.path.join(root, mode, output_dirs[0]))  # 保存 Canny 处理的图像

        
        dilated_edges1 = dilate_edges(edges)
        
        if not os.path.exists(os.path.join(root, mode, output_dirs[1])):
            os.makedirs(os.path.join(root, mode, output_dirs[1]))
        edge_filename = os.path.join(root, mode, output_dirs[1], data_list[i])
        print(edge_filename)
        cv.imwrite(edge_filename, dilated_edges1)  # 保存 Canny + Dilate 处理的图像




root = 'D:\djy\levir'

mode = ['train', 'val', 'test']
output_dirs = ['edge_slim', 'edge']

for n in mode:
    data_list, _ = read_images(root, n)
    process_images(root, n, output_dirs)