import os
import numpy as np
from torch.utils import data
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch




def read_images(root,mode,edgename):#读取文件list名字  
   
    img_A_dir = os.path.join(root, mode, 'A')
    img_B_dir = os.path.join(root, mode, 'B')
    label_dir = os.path.join(root, mode, 'label')
    
    
    edge_dir = os.path.join(root, mode, edgename)
        

    data_list = os.listdir(img_A_dir)#因为用了同样一个图片同样的读取方式因此只用list一个就行
    imgs_list_A, imgs_list_B, labels= [], [], []
    edges=[]

    print('%s image loading' %mode)
    for it in tqdm(data_list):
        # print(it)
        if (it[-4:]=='.png' or it[-4:]=='.jpg'):
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)
            label_path = os.path.join(label_dir, it)
            
            
        
            imgs_list_A.append(img_A_path)
            imgs_list_B.append(img_B_path)
            labels.append(label_path)
            
            edge_path = os.path.join(edge_dir, it)
            edges.append(edge_path)
                
    return imgs_list_A, imgs_list_B, labels,edges
    

class Data(data.Dataset):
    def __init__(self,root, mode,edgename="edge", data_enhancement = False):#这里设计上一定要有edge文件加
        self.data_enhancement = data_enhancement
        
        
        self.imgs_list_A, self.imgs_list_B, self.labels,self.edges = read_images(root,mode,edgename)
  
        
        
        self.transform2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#灰度图是0.5，0.5
            ])
        self.transform3 = transforms.Compose([
                                              transforms.ToTensor(),
                                              
           ])

        
        
        
    


    def __getitem__(self, idx):
        
        img_A = Image.open(self.imgs_list_A[idx])
        img_B = Image.open(self.imgs_list_B[idx])
        
        
        edge = Image.open(self.edges[idx])
        label = Image.open(self.labels[idx])
        
        
        p1 = np.random.choice([0, 1])
        p2 = np.random.choice([0, 1])
        transform1 = transforms.Compose([
            transforms.RandomHorizontalFlip(p1),
            transforms.RandomVerticalFlip(p2),
            #transforms.Resize([256,256]),
        ])
        
        if self.data_enhancement:
            img_A=transform1(img_A)
            img_B=transform1(img_B)
            
            
            edge=transform1(edge)

            label=transform1(label)

        img_A=self.transform2(img_A)
        img_B=self.transform2(img_B)
        
        
        edge=self.transform3(edge)
            
        
        label=self.transform3(label)
        #label=torch.where(label>0,1,0).float()这里不用绝对的是因为图像经过了缩小 出现10之间的信息能保存更多信息
        
            
        return img_A, img_B,edge,label


    def __len__(self):
        return len(self.imgs_list_A)
    
