import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', stride=1):
        super().__init__()
        self.ActivationLayer = nn.LeakyReLU(inplace=True)
        if activation == 'relu':
            self.ActivationLayer = nn.ReLU(inplace=True)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            self.ActivationLayer,
        )

    def forward(self, x):
        return self.conv_relu(x)


class ConvTranspose2D(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.ActivationLayer = nn.LeakyReLU(inplace=True)
        if activation == 'relu':
            self.ActivationLayer = nn.ReLU(inplace=True)
        self.deconv_relu = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            self.ActivationLayer,
        )

    def forward(self, x):
        return self.deconv_relu(x)


class MaxPooling2D(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.maxpool(x)


class AvgPooling2D(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.avgpool(x)


class Concat(nn.Module):
    def forward(self, x, y):
        _, _, xh, xw = x.size()
        _, _, yh, yw = y.size()
        diffY = xh - yh
        diffX = xw - yw
        y = F.pad(y, (diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2))
        return torch.cat((x, y), dim=1)


def conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    padding = kernel_size // 2 if dilation == 1 else dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, dilation=dilation)
    # return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, dilation=dilation)


class SASBlock(nn.Module):
    def __init__(self, n_view, n_feats):
        super(SASBlock, self).__init__()

        self.an = n_view
        self.relu = nn.ReLU(inplace=True)
        self.spaconv = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1)
        self.angconv = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1)

        # self.spaconv = ResBlock(n_feats=n_feats, kernel_size=3)
        # self.angconv = ResBlock(n_feats=n_feats, kernel_size=3)

    def forward(self, x):
        N, c, h, w = x.shape  # [N*an2,c,h,w]
        N = N // (self.an * self.an)

        out = self.relu(self.spaconv(x))  # [N*an2,c,h,w]
        out = out.reshape(N, self.an * self.an, c, h * w)
        out = torch.transpose(out, 1, 3)
        out = out.reshape(N * h * w, c, self.an, self.an)  # [N*h*w,c,an,an]

        out = self.relu(self.angconv(out))  # [N*h*w,c,an,an]
        out = out.reshape(N, h * w, c, self.an * self.an)
        out = torch.transpose(out, 1, 3)
        out = out.reshape(N * self.an * self.an, c, h, w)  # [N*an2,c,h,w]
        return out


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), dilation=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias, dilation=dilation))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        # self.se = SELayer(n_feats, reduction=8)
        # self.conv_skip = nn.Conv2d(n_feats, n_feats, 1)

    def forward(self, x):
        res = self.body(x)
        # res = self.se(res)
        res += x

        return res


class ResBlock3D(nn.Module):
    def __init__(self, n_feats, kernel_size, dilation=None):
        super(ResBlock3D, self).__init__()
        if dilation:
            padding = dilation
        else:
            padding = tuple((i // 2 for i in kernel_size))
            dilation = (1, 1, 1)

        self.body = nn.Sequential(
            nn.Conv3d(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, bias=True,
                      padding=padding, dilation=dilation),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, bias=True,
                      padding=padding, dilation=dilation)
        )

    def forward(self, x):
        res = self.body(x)
        # print(res.shape)
        res += x
        return res


class AM(nn.Module):
    def __init__(self, n_feats):
        super(AM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=2, dilation=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=3, dilation=3),
            nn.ReLU()
        )
        self.att = ChannelAttention(in_planes=n_feats * 3)
        self.agg = nn.Conv2d(in_channels=n_feats * 3, out_channels=n_feats, kernel_size=1, bias=False)
        self.out = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            # nn.ReLU()
        )

    def forward(self, x):
        fea_1 = self.conv1(x)
        fea_2 = self.conv2(x)
        fea_3 = self.conv3(x)

        feas = torch.cat([fea_1, fea_2, fea_3], dim=1)
        att = self.att(feas)

        feas_ = feas * att
        feas = feas + feas_
        agg = self.agg(feas)

        out = self.out(agg)

        return out + x



class ResASPP(nn.Module):
    def __init__(self, n_feats=32):
        super(ResASPP, self).__init__()
        self.head_1 = nn.Sequential(
            ASPPConv(in_channels=n_feats, out_channels=n_feats, dilation=1),
            ResBlock(n_feats=n_feats, kernel_size=3, dilation=1)
        )
        self.head_2 = nn.Sequential(
            ASPPConv(in_channels=n_feats, out_channels=n_feats, dilation=2),
            ResBlock(n_feats=n_feats, kernel_size=3, dilation=2)
        )
        self.head_3 = nn.Sequential(
            ASPPConv(in_channels=n_feats, out_channels=n_feats, dilation=3),
            ResBlock(n_feats=n_feats, kernel_size=3, dilation=3)
        )

        self.out = nn.Sequential(
            nn.Conv2d(3 * n_feats, n_feats, 1, bias=False),
        )

    def forward(self, x):
        res_list = [self.head_1(x), self.head_2(x), self.head_3(x)]
        res = torch.cat(res_list, dim=1)

        return self.out(res)


class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        if activation == 'relu':
            self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.8)
        self.cbam = CBAM(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.8)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        x1 = self.relu(bn1)
        cbam = self.cbam(x1)
        conv2 = self.conv2(cbam)
        bn2 = self.bn1(conv2)
        out = bn2 + x
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        # maxout = self.sharedMLP(self.max_pool(x))
        # return self.sigmoid(avgout + maxout)
        return self.sigmoid(avgout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        out = self.sa(x) * x
        return out


def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    downscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]
    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c, 1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)


class PixelUnShuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        downscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''
        return pixel_unshuffle(input, self.downscale_factor)


class MSASBlock(nn.Module):
    def __init__(self, n_view, n_feats=32):
        super(MSASBlock, self).__init__()

        self.an = n_view
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.down = PixelUnShuffle(2)

        self.angular_conv = nn.Sequential(
            nn.Conv2d(in_channels=n_feats*4, out_channels=n_feats*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up = nn.PixelShuffle(2)

        # self.spaconv = ResBlock(n_feats=n_feats, kernel_size=3)
        # self.angconv = ResBlock(n_feats=n_feats, kernel_size=3)

    def forward(self, x):
        N, c, h, w = x.shape  # [N*an2,c,h,w]
        N = N // (self.an * self.an)

        out = self.spatial_conv(x)  # [N*an2,c,h,w]
        out = self.down(out)
        out = out.reshape(N, self.an * self.an, c * 4, h * w // 4)

        out = torch.transpose(out, 1, 3)
        out = out.reshape(N * h * w // 4, c * 4, self.an, self.an)  # [N*h*w,c,an,an]

        out = self.angular_conv(out)  # [N*h*w,c,an,an]
        out = out.reshape(N, h * w // 4, c * 4, self.an * self.an)
        out = torch.transpose(out, 1, 3)
        out = out.reshape(N * self.an * self.an, c * 4, h * w // 4)  # [N*an2,c,h,w]
        out = out.reshape(N * self.an * self.an, c * 4, h // 2, w // 2)  # [N*an2,c,h,w]
        out = self.up(out)

        return out



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


