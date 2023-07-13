import os.path
import torch
from data.image_folder import make_dataset
from data.preprocessing import Preprocessing
from PIL import Image
import numpy as np


class ChangeDetectionDataset(torch.utils.data.Dataset):

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input T1_img 
        dir_t1 = 'A'
        self.dir_t1 = os.path.join(opt.dataroot, opt.dataset, opt.phase, dir_t1)
        self.t1_paths = sorted(make_dataset(self.dir_t1))

        ### input T2_img
        dir_t2 = 'B'
        self.dir_t2 = os.path.join(opt.dataroot, opt.dataset, opt.phase, dir_t2)
        self.t2_paths = sorted(make_dataset(self.dir_t2))

        ### input change_label
        dir_label = 'label'
        self.dir_label = os.path.join(opt.dataroot, opt.dataset, opt.phase, dir_label)
        self.label_paths = sorted(make_dataset(self.dir_label))

        self.dataset_size = len(self.t1_paths)
        if self.opt.phase == 'train':
            self.preprocess = Preprocessing(
                                            img_size=self.opt.img_size,
                                            with_random_hflip=opt.aug,
                                            with_random_vflip=opt.aug,
                                            with_scale_random_crop=opt.aug,
                                            with_random_blur=opt.aug,
                                            )
        else:
            self.preprocess= Preprocessing(
                                            img_size=self.opt.img_size,
                                            )

    def __getitem__(self, index):
        ### input T1_img 
        t1_path = self.t1_paths[index]
        t1_img = np.asarray(Image.open(t1_path).convert('RGB'))

        ### input T2_img
        t2_path = self.t2_paths[index]
        t2_img = np.asarray(Image.open(t2_path).convert('RGB'))

        ### input label
        label_path = self.label_paths[index]
        label = np.array(Image.open(label_path), dtype=np.uint8)
        if self.opt.label_norm == True:
            label = label // 255

        ### transform
        [t1_tensor, t2_tensor], [label_tensor] = self.preprocess.transform([t1_img, t2_img], [label], to_tensor=True)

        input_dict = {'t1_img': t1_tensor, 't2_img': t2_tensor, 'label': label_tensor,
                      't1_path': t1_path, 't2_path': t2_path, 'label_path': label_path}

        return input_dict

    def __len__(self):
        return len(self.t1_paths) // self.opt.batch_size * self.opt.batch_size
