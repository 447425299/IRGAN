import os.path
import random
from data.base_dataset import BaseDataset, get_simple_transform
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        #EDGEA
        # pathA = opt.dataroot + '/' + opt.phase +'edge'
        # self.dir_edgeA = os.path.join(pathA)
        # self.edgeA_paths = sorted(make_dataset(self.dir_edgeA))
        #instance
        # pathinstance = opt.dataroot + '/' + 'instance'
        # self.pathtxt = opt.dataroot + '/' + 'annotation.txt'
        # self.dir_instance = os.path.join(pathinstance)
        # self.instance_path = sorted(make_dataset(self.dir_instance))

        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')  # only support this mode
        assert(self.opt.loadSize >= self.opt.fineSize)
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.transform_A = get_simple_transform(grayscale=(input_nc == 1))
        self.transform_B = get_simple_transform(grayscale=(output_nc == 1))
        self.transform_instance = get_simple_transform(grayscale=(input_nc == 1))

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        # data = pd.read_table(self.pathtxt, header=None, delim_whitespace=True)
        # x, y = data.loc[index, 1], data.loc[index, 2]
        # gt = np.zeros((257, 385))
        # gt[y, x] = 1.0
        #edgeA读取
        # edgeA_path = AB_path[:(len(AB_path)-11)] + 'edge' +AB_path[(len(AB_path)-11):(len(AB_path)-4)] + '_edge.png'
        # edgeA = Image.open(edgeA_path).convert('RGB')
        # instance读取
        # instance_path = self.instance_path[index]
        # instance = Image.open(instance_path).convert('RGB')

        w, h = AB.size
        w2 = int(w / 2)
        A0 = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        B0 = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        x, y, h, w = transforms.RandomCrop.get_params(A0, output_size=[self.opt.fineSize, self.opt.fineSize])
        A = TF.crop(A0, x, y, h, w)
        B = TF.crop(B0, x, y, h, w)
        # edgeA = TF.crop(edgeA, x, y, h, w)

        if (not self.opt.no_flip) and random.random() < 0.5:
            A = TF.hflip(A)
            B = TF.hflip(B)
        A = self.transform_A(A)
        B = self.transform_B(B)
        # instance = A[:,x:x+256, y:y+256]#self.transform_instance(instance)
        # edgeA = self.transform_edgeA(edgeA)
        # B = torch.cat((B, edgeA), dim=0)
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}#, 'instance': instance, 'instance_path': instance_path, 'gt': gt}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
