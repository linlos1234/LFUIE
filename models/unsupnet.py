import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from torch.autograd import Variable
from math import floor
import math


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(fn, fn, 3, padding=1)
        self.conv2 = nn.Conv2d(fn, fn, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(fn)
        self.norm2 = nn.BatchNorm2d(fn)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        return identity + out

def make_layer(block, p, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block(p))
    return nn.Sequential(*layers)



class UNet(nn.Module):
    def __init__(self, in_ch):
        super(UNet, self).__init__()
        self.ch1 = nn.Conv2d(in_ch, 64, 3, 1, 1)
        self.conv1 = make_layer(ResidualBlock, 64, 2)
        self.pool1 = nn.MaxPool2d(2)

        self.ch2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2 = make_layer(ResidualBlock, 128, 2)
        self.pool2 = nn.MaxPool2d(2)

        self.ch3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3 = make_layer(ResidualBlock, 256, 2)
        self.pool3 = nn.MaxPool2d(2)

        self.ch4 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4 = make_layer(ResidualBlock, 512, 2)


        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.ch7 = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv7 = make_layer(ResidualBlock, 256, 2)
        self.head7_d = nn.Conv2d(256, 1, 1)
        self.head7_c = nn.Conv2d(256, 1, 1)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.ch8 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv8 = make_layer(ResidualBlock, 128, 2)
        self.head8_d = nn.Conv2d(128, 1, 1)
        self.head8_c = nn.Conv2d(128, 1, 1)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.ch9 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv9 = make_layer(ResidualBlock, 64, 2)
        self.head9_d = nn.Conv2d(64, 1, 1)
        self.head9_c = nn.Conv2d(64, 1, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(x.shape)
        c1 = self.conv1(self.relu(self.ch1(x)))
        p1 = self.pool1(c1)
        c2 = self.conv2(self.relu(self.ch2(p1)))
        p2 = self.pool2(c2)
        c3 = self.conv3(self.relu(self.ch3(p2)))
        p3 = self.pool3(c3)
        c4 = self.conv4(self.relu(self.ch4(p3)))
        # p4 = self.pool4(c4)
        # c5 = self.conv5(p4)
        # up_6 = self.up6(c5)
        # merge6 = torch.cat([up_6, c4], dim=1)
        # c6 = self.conv6(merge6)
        
        # print(c4.shape)
        
        up_7 = self.up7(c4)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(self.relu(self.ch7(merge7)))
        disp7 = self.head7_d(c7)
        conf7 = self.head7_c(c7)
        out7 = [disp7, conf7]
        
        # print(c7.shape)

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(self.relu(self.ch8(merge8)))
        disp8 = self.head8_d(c8)
        conf8 = self.head8_c(c8)
        out8 = [disp8, conf8]

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(self.relu(self.ch9(merge9)))
        disp9 = self.head9_d(c9)
        conf9 = self.head9_c(c9)
        out9 = [disp9, conf9]

        return out9



class DispNet(nn.Module):
    def __init__(self, n_view):
        super(DispNet, self).__init__()
        sub_lf_num = math.ceil(n_view / 2) ** 2
        self.unet = UNet(in_ch=sub_lf_num*3)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lf):
        # N, an2, c, h, w
        n, u, v, c, h, w = lf.shape
        subLFs = lf2sublfs(lf.reshape(n,-1,c,h,w))
        N, _, an2, c, h, w = subLFs.shape # [N,4,an2,c,h,w]

        out = self.unet(subLFs.reshape(N*4, an2*c, h, w))
        # print(len(out))

        disp_init_sub, conf_init = out

        disp_init_sub = disp_init_sub.view(N, 4, h, w)
        conf_init = torch.sigmoid(conf_init.view(N, 4, h, w))

        disp_fliped = sub_spatial_flip(disp_init_sub)
        conf_fliped = sub_spatial_flip(conf_init)
        conf_fliped_norm = self.softmax(conf_fliped)

        disp_init = torch.sum(disp_fliped*conf_fliped_norm, dim=1).unsqueeze(1) #[N,1,h,w]

        return disp_fliped, conf_fliped_norm, disp_init


def lf2sublfs(lf):
    # print('lf', lf.shape)
    N, an2, c, h, w = lf.shape
    an = int(math.sqrt(an2))

    ind_cv = (an - 1) // 2
    # print('ind cv', ind_cv)

    lf_lu = lf.view(N, an, an, c, h, w)
    lf_ru = torch.flip(lf_lu, [2, 5])
    lf_ld = torch.flip(lf_lu, [1, 4])
    lf_rd = torch.flip(lf_lu, [1, 2, 4, 5])

    sub_lf_lu = lf_lu[:, :ind_cv+1, :ind_cv+1].contiguous().view(N, -1, c, h, w)
    sub_lf_ru = lf_ru[:, :ind_cv+1, :ind_cv+1].contiguous().view(N, -1, c, h, w)
    sub_lf_ld = lf_ld[:, :ind_cv+1, :ind_cv+1].contiguous().view(N, -1, c, h, w)
    sub_lf_rd = lf_rd[:, :ind_cv+1, :ind_cv+1].contiguous().view(N, -1, c, h, w)

    sub_lfs = torch.stack([sub_lf_lu, sub_lf_ru, sub_lf_ld, sub_lf_rd], 1)  # [N,4,an,3,h,w]

    return sub_lfs

def sub_spatial_flip(sub_imgs):
    # [N, 4, h, w]
    sub_imgs = torch.stack([sub_imgs[:, 0],
                            torch.flip(sub_imgs[:, 1], [2]),
                            torch.flip(sub_imgs[:, 2], [1]),
                            torch.flip(sub_imgs[:, 3], [1, 2]),
                            ], 1)
    return sub_imgs

def sub_spatial_flip_5D(sub_imgs):
    # [N, 4, c, h, w]
    sub_imgs = torch.stack([sub_imgs[:, 0],
                            torch.flip(sub_imgs[:, 1], [3]),
                            torch.flip(sub_imgs[:, 2], [2]),
                            torch.flip(sub_imgs[:, 3], [2, 3]),
                            ], 1)
    return sub_imgs