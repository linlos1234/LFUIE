# MSPNet with more pattern and multi-scale 
import torch
import torch.nn as nn
import torch.nn.functional as F
from unsupnet import DispNet
import math
import numpy as np
from base_layers import *

class Conv_spa(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, bias):
        super(Conv_spa, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride = stride, padding = padding, bias = bias),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):     # [N,32,uv,h,w]
        N,u,v,c,h,w = x.shape
        x = x.reshape(N*u*v,c,h,w)  # [N*uv,32,h,w]
        out = self.op(x)
        #print(out.shape)
        out = out.reshape(N,u,v,32,h,w)
        return out

class Conv_ang(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, angular, bias):
        super(Conv_ang, self).__init__()
        self.angular = angular
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride = stride, padding = padding, bias = bias),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):    # [N,32,uv,h,w]
        N,u,v,c,h,w = x.shape
        x = x.permute(0,4,5,3,1,2).reshape(N*h*w,c,self.angular,self.angular)   #[N*h*w,32,7,7]
        out = self.op(x)
        out = out.reshape(N,h,w,32,u,v).permute(0,4,5,3,1,2)
        return out
    
class Conv_epi_h(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, bias):
        super(Conv_epi_h, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride = stride, padding = padding, bias = bias),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):    # [N,64,uv,h,w]
        N,u,v,c,h,w = x.shape
        x = x.permute(0,1,4,3,2,5).reshape(N*u*h,c,v,w)
        out = self.op(x)
        out = out.reshape(N,u,h,32,v,w).permute(0,1,4,3,2,5)
        return out

class Conv_epi_v(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, bias):
        super(Conv_epi_v, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride = stride, padding = padding, bias = bias),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):
        N,u,v,c,h,w = x.shape
        x = x.permute(0,2,5,3,1,4).reshape(N*v*w,c,u,h)
        out = self.op(x)
        out = out.reshape(N,v,w,32,u,h).permute(0,4,1,3,5,2)
        return out

class Autocovnlayer(nn.Module):
    def __init__(self,angular,fn=32,bs=True):
        super(Autocovnlayer, self).__init__()
        self.angular = angular
        self.kernel_size = 3

        self.naslayers = nn.ModuleList([
           Conv_spa(C_in = fn, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs),
           Conv_ang(C_in = fn, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, angular = self.angular, bias = bs),
           Conv_epi_h(C_in = fn, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs),
           Conv_epi_v(C_in = fn, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs)
        ])
        
        self.epi_boost = nn.Conv2d(in_channels = fn, out_channels=32, kernel_size=3, stride=1, padding=1, bias = bs)
        self.Conv_mixnas = nn.Conv2d(in_channels = 32*5, out_channels=32, kernel_size=1, stride=1, padding=0, bias = False)     ## 1*1 paddding!!
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        [N, u, v, _, h, w] = x.shape 
        nas = []
        for layer in self.naslayers:
            nas_ = layer(x)
            nas.append(nas_)
        x_epi = nas[-1] + nas[-2]    # (N,u,v,32,h,w)
        nas_ = self.relu(self.epi_boost(x_epi.reshape(N*u*v,-1,h,w)))
        nas.append(nas_.reshape(N,u,v,-1,h,w))
        
        nas = torch.stack(nas,dim = 0)   
        nas = nas.permute([1,2,3,0,4,5,6]).reshape([N*u*v,5*32,h,w])   ##[N*uv, 5*fn, h,w] 
        out = self.relu(self.Conv_mixnas(nas))
        out = out.reshape(N,u,v,-1,h,w)           
        return out


class DynamicDWConv(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, groups=1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.groups = groups

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        Block1 = [nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2, groups=channels)
                  for _ in range(3)]
        Block2 = [nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2, groups=channels)
                  for _ in range(3)]
        self.tokernel = nn.Conv2d(channels, kernel_size ** 2 * self.channels, 1, 1, 0)
        self.bias = nn.Parameter(torch.zeros(channels))
        self.Block1 = nn.Sequential(*Block1)
        self.Block2 = nn.Sequential(*Block2)

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.tokernel(self.pool(self.Block2(self.maxpool(self.Block1(self.avgpool(x))))))
        weight = weight.view(b * self.channels, 1, self.kernel_size, self.kernel_size)
        # x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding, groups=b * self.groups)
        # x = x.view(b, c, x.shape[-2], x.shape[-1])
        return weight

class Autocovnlayer_dynamic(nn.Module):
    def __init__(self,angular,fn=32,bs=False):
        super(Autocovnlayer_dynamic, self).__init__()
        self.angular = angular
        self.kernel_size = 3

        self.naslayers = nn.ModuleList([
           Conv_spa(C_in = fn, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs),
           Conv_ang(C_in = fn, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, angular = self.angular, bias = bs),
           Conv_epi_h(C_in = fn, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs),
           Conv_epi_v(C_in = fn, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs)
        ])
        
        self.epi_boost = nn.Conv2d(in_channels = fn, out_channels=32, kernel_size=3, stride=1, padding=1, bias = bs)
        self.Conv_mixnas = nn.Conv2d(in_channels = 32*2, out_channels=32, kernel_size=1, stride=1, padding=0, bias = False)     ## 1*1 paddding!!
        self.relu = nn.ReLU(inplace=True)

        self.dyd = DynamicDWConv(self.angular**2 * fn, 3, 1, self.angular**2 * fn)
        self.bais = nn.Parameter(torch.zeros(self.angular**2 * fn))

    def forward(self,x):
        [N, u, v, _, h, w] = x.shape 
        nas = []
        for layer in self.naslayers:
            nas_ = layer(x)
            nas.append(nas_)
        x_epi = nas[-1] + nas[-2]    # (N,u,v,32,h,w)

        core_weight = self.dyd(x_epi.reshape(N,-1,h,w)) #[1,u*v*32]
        m1 = F.conv2d(nas[0].reshape(N,-1,h,w), core_weight, self.bais.repeat(1), stride=1, padding=1, groups=self.angular**2 * 32)
        m1 = m1.reshape(N,u,v,-1,h,w)
        m2 = F.conv2d(nas[1].reshape(N,-1,h,w), core_weight, self.bais.repeat(1), stride=1, padding=1, groups=self.angular**2 * 32)
        m2 = m2.reshape(N,u,v,-1,h,w)
        
        nas = torch.stack([m1, m2],dim = 0)   
        nas = nas.permute([1,2,3,0,4,5,6]).reshape([N*u*v,2*32,h,w])   ##[N*uv, 2*fn, h,w] 
        out = self.relu(self.Conv_mixnas(nas))
        out = out.reshape(N,u,v,-1,h,w)           
        return out

class Dgfm(nn.Module):
    def __init__(self, channels=32):
        super(Dgfm, self).__init__()
        #self.interpolate = nn.functional.interpolate
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, padding=1)   
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x_init, d):
        tmp = self.relu1(self.conv1(d))
        gamma = self.conv2(tmp)
        # beta = self.conv3(tmp)
        x = x_init * gamma + x_init
        return x

class Stage1(nn.Module):
    def __init__(self, n_view=5,fn=32):
        super(Stage1, self).__init__()

        self.conv_init = nn.Sequential(
            nn.Conv2d(3, fn, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.nas_up = Autocovnlayer(n_view,fn)
        self.nas_down = Autocovnlayer(n_view,fn)
        self.down0 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.down = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.conv_low_cat = nn.Sequential(
            nn.Conv2d(32 * 2, 32 * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResBlock(32 * 2, kernel_size=3),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(fn * 2, fn, kernel_size=2, stride=2)
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(fn * 2, fn * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResBlock(fn * 2, kernel_size=3),
            nn.Conv2d(fn * 2, fn, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.nas_conv1 = Autocovnlayer(n_view,fn)
        self.nas_conv2 = Autocovnlayer(n_view,fn)
        self.nas_conv3 = Autocovnlayer_dynamic(n_view,fn)
        # self.out = nn.Sequential(
        #     nn.Conv2d(fn, 3, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ReLU(),
        # )
        self.out = nn.Conv2d(fn, 3, kernel_size=3, stride=1, padding=1, bias=False)

        self.re0 = Dgfm()

    def forward(self, x_init, d):
        N, u, v, c, h, w = x_init.shape
        x = self.conv_init(x_init.reshape(N*u*v,-1,h,w))
        x_low = self.down0(x.reshape(N*u*v,-1,h,w))
        feats_low = self.nas_down(x_low.reshape(N,u,v,-1,x_low.size(-2),x_low.size(-1)))
        feats = self.nas_up(x.reshape(N,u,v,-1,h,w))
        down = self.down(feats.reshape(N*u*v,-1,h,w))

        low_cat = torch.cat([feats_low.reshape(N*u*v,-1,x_low.size(-2),x_low.size(-1)), down], dim=1)
        low_cat = self.conv_low_cat(low_cat)
        up = self.up(low_cat)
        high_cat = torch.cat([feats.reshape(N*u*v,-1,h,w), up], dim=1)
        feats = self.conv_last(high_cat)
        feats = feats.reshape(N,u,v,-1,h,w)
        feats = self.nas_conv1(feats)+feats
        feats = self.nas_conv2(feats)+feats
        feats = self.nas_conv3(feats)+feats

        out = self.out(self.re0(feats.reshape(N*u*v,-1,h,w), d))
        out = out.reshape(N,u,v,-1,h,w)
        return feats, out

class Stage2(nn.Module):
    def __init__(self, n_view=5,fn=32):
        super(Stage2, self).__init__()
        self.conv_init = nn.Sequential(
            nn.Conv2d(3, fn, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.nas = Autocovnlayer(n_view,fn)
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(fn*2, fn, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.nas_conv1 = Autocovnlayer(n_view,fn)
        self.nas_conv2 = Autocovnlayer(n_view,fn)
        self.nas_conv3 = Autocovnlayer_dynamic(n_view,fn)

        # self.out = nn.Sequential(
        #     nn.Conv2d(fn, 3, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ReLU(),
        # )
        self.re0 = Dgfm()
        self.out = nn.Conv2d(fn, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, feats_pre, x_init, d):
        N, u, v, c, h, w = x_init.shape
        x = self.conv_init(x_init.reshape(N*u*v,-1,h,w))
        feats_fusion = self.nas(x.reshape(N,u,v,-1,h,w))
        feats = torch.cat([feats_pre, feats_fusion],dim=3)
        feats = self.conv_fuse(feats.reshape(N*u*v,-1,h,w))

        feats = feats.reshape(N,u,v,-1,h,w)
        feats = self.nas_conv1(feats)+feats
        feats = self.nas_conv2(feats)+feats
        feats = self.nas_conv3(feats)+feats

        out = self.out(self.re0(feats.reshape(N*u*v,-1,h,w), d))
        out = out.reshape(N,u,v,-1,h,w)
        return feats, out

class Stage3(nn.Module):
    def __init__(self, n_view=5, fn=32):
        super(Stage3, self).__init__()
        self.conv_init = nn.Sequential(
            nn.Conv2d(3, fn, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.nas = Autocovnlayer(n_view,fn)
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(fn*2, fn, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.nas_conv1 = Autocovnlayer(n_view,fn)
        self.nas_conv2 = Autocovnlayer(n_view,fn)
        self.nas_conv3 = Autocovnlayer_dynamic(n_view,fn)

        # self.out = nn.Sequential(
        #     nn.Conv2d(fn, 3, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ReLU(),
        # )
        self.re0 = Dgfm()
        self.out = nn.Conv2d(fn, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, feats_pre, x_init, d):
        N, u, v, c, h, w = x_init.shape
        x = self.conv_init(x_init.reshape(N*u*v,-1,h,w))
        feats_fusion = self.nas(x.reshape(N,u,v,-1,h,w))
        feats = torch.cat([feats_pre, feats_fusion],dim=3)
        feats = self.conv_fuse(feats.reshape(N*u*v,-1,h,w))

        feats = feats.reshape(N,u,v,-1,h,w)
        feats = self.nas_conv1(feats)+feats
        feats = self.nas_conv2(feats)+feats
        feats = self.nas_conv3(feats)+feats

        out = self.out(self.re0(feats.reshape(N*u*v,-1,h,w), d))
        out = out.reshape(N,u,v,-1,h,w)
        return out

class Net(nn.Module):
    def __init__(self, n_view=5):
        super(Net, self).__init__()

        self.stage1 = Stage1(n_view)
        self.stage2 = Stage2(n_view)
        self.stage3 = Stage3(n_view)
        
        self.dispnet = DispNet(n_view)
        self.dispnet.cuda()
        model = torch.load('./weights/unsupnet_7.pth')
        self.dispnet.load_state_dict(model['model'])

    def forward(self, x):
        
        with torch.no_grad():
            _,_,d = self.dispnet(x)
        feats, out1 = self.stage1(x, d)
        with torch.no_grad():
            _,_,d = self.dispnet(out1)
        feats, out2 = self.stage2(feats, out1, d)
        with torch.no_grad():
            _,_,d = self.dispnet(out2)
        out3 = self.stage3(feats, out2, d)

        return out1, out2, out3


