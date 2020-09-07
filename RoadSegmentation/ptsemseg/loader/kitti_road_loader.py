import os
import torch
import numpy as np
import cv2

from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import *
import random
import collections
from sklearn.utils import shuffle

class KITTIRoadLoader(data.Dataset):
    """KITTI Road Dataset Loader

    http://www.cvlibs.net/datasets/kitti/eval_road.php

    Data is derived from KITTI
    """
    mean_rgb = [103.939, 116.779, 123.68] # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(self, root, split="training", is_transform=False,img_norm = True, 
                 img_size=(1280, 384), augmentations=None, version='pascal', phase='train'):
        """__init__

        :param root:
        :param split:
        :param is_transform: (not used)
        :param img_size: (not used)
        :param augmentations  (not used)
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 2
        self.img_size = img_size 
        self.mean = np.array(self.mean_rgb)
        self.img_norm = img_norm

        self.files = collections.defaultdict(list)
        
        files_list = os.listdir(self.root+'/'+"training"+"/image_2")
        shuffle(files_list, random_state=42)
        n = int(0.8*len(files_list))
        if (self.split == 'training'):
          
          self.files[self.split] = files_list[:n]
        elif(self.split == 'val'):
          self.files[self.split] = files_list[n:]

        

        # if phase == 'train':
        #     self.images_base = os.path.join(self.root, 'training', 'image_2')
        #     #self.lidar_base = os.path.join(self.root, 'training', 'ADI')
        #     self.annotations_base = os.path.join(self.root, 'training', 'gt_image_2')
        #     files_list = os.listdir(self.images_base)
        #     self.im_files = recursive_glob(rootdir=self.images_base, suffix='.png')
        # else:
        #     self.images_base = os.path.join(self.root, 'testing', 'image_2')
        #     #self.lidar_base = os.path.join(self.root, 'testing', 'ADI')
        #     self.annotations_base = os.path.join(self.root, 'testing', 'gt_image_2')
        #     self.split = 'test'

        #     self.im_files = recursive_glob(rootdir=self.images_base, suffix='.png')
        #     self.im_files = sorted(self.im_files)

        self.phase = phase

        #print("Found %d %s images" % (self.data_size, self.split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_name = self.files[self.split][index]
        img_path = self.root + "/" + "training" + "/image_2/" + img_name
        im_name_splits = img_path.split(os.sep)[-1].split('.')[0].split('_')

        img = cv2.imread(img_path)
        img = np.array(img, dtype=np.uint8)        

        
        annotations_base = self.root + "/"+"training"+"/gt_image_2"
        # if im_name_splits[0] == "uu" or im_name_splits[0] == "umm":
        #   add_path = '_road_'
        # elif im_name_splits[0] == 'um':
        #   add_path = '_lane_'
        lbl_path = os.path.join(annotations_base,im_name_splits[0] + "_road_" + im_name_splits[1] + '.png')

        lbl_tmp = cv2.imread(lbl_path)
        lbl_tmp = np.array(lbl_tmp, dtype=np.uint8)
            
        # lbl = 255 + np.zeros( (img.shape[0], img.shape[1]), np.uint8)
        # lbl[lbl_tmp[:,:,0] > 0] = 1
        # lbl[(lbl_tmp[:,:,2] > 0) & (lbl_tmp[:,:,0] == 0)] = 0
        lbl = lbl_tmp[:,:,0]
        
        if self.augmentations is not None:
          img, lbl = self.augmentations(img, lbl)  
        img, lbl = self.transform(img, lbl)
        
        

        return img, lbl
        


    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
       
      
        

       
        lbl = cv2.resize(lbl, (int(self.img_size[1]), int(self.img_size[0])), interpolation=cv2.INTER_NEAREST)
        lbl = lbl.astype('float')/255.0
        lbl = torch.from_numpy(lbl).long()
        return img, lbl
        
        

