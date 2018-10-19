from collections import OrderedDict

import pretrainedmodels
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from torchvision import models
import torchvision
from torchvision.models import ResNet
from torchvision.models.resnet import model_urls, BasicBlock

from src.config import *
from src.utils import load_checkpoint
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(checkpoint=None):
    model = AttentionUnet()
    state = {'epoch': 0, 'lb_acc': 0}
    if LOAD_CHECKPOINT:
        state = load_checkpoint(model, checkpoint)

    # model.freeze_encoder()
    optimizer = torch.optim.SGD(model.trainable_params(), lr=LEARNING_RATE, weight_decay=L2_REG)

    # optimizer = torch.optim.Adam(model.trainable_params(), lr=LEARNING_RATE, weight_decay=L2_REG, amsgrad=True)
    model.train()
    model.to(DEVICE)
    scheduler = ReduceLROnPlateau(mode='max', optimizer=optimizer, min_lr=1e-3,
                                  patience=DECREASE_LR_EPOCH, factor=0.5, verbose=True)
    print("Trainable Parameters: %s" % count_parameters(model))

    return model, optimizer, scheduler, state


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ConvBnRelu(nn.Module):
    def __init__(self, in_, out, bn=True):
        super().__init__()
        self.bn = bn
        self.conv = conv3x3(in_, out)
        if self.bn:
            self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class unetUp(nn.Module):
    def __init__(self, in_, skip_, middle_, out_):
        super(unetUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_, in_, kernel_size=4, stride=2, padding=1)
        self.conv1 = unetConv2(in_ + skip_, middle_, is_batchnorm=True)
        self.conv2 = unetConv2(middle_, out_, is_batchnorm=True)
        self.scse = SELayer(out_)

    def forward(self, inputs, skip):
        outputs2 = self.up(inputs)

        offset = outputs2.size()[2] - skip.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(skip, padding, mode='replicate')
        catted = torch.cat([outputs1, outputs2], 1)

        x = self.conv1(catted)
        x = self.conv2(x)
        x = self.scse(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # print(x.size())
        return x.contiguous().view(x.size(0), -1)


class UnetDsv(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv, self).__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='bilinear'))

    def forward(self, input):
        return self.dsv(input)


class AttentionUnet(nn.Module):

    def __init__(self, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(AttentionUnet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm

        encoder_outputs = [64, 256, 512, 1024, 2048]
        filter_param = 64

        decoder_sizes = [filter_param, filter_param, filter_param, filter_param, filter_param]
        middles = [96, 96, 128, 128, 128]

        self.encoder = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet')

        # self.conv1 = nn.Sequential(*list(self.encoder.layer0)[:-1])  # Drop last pooling layer

        layer0_modules = [
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=1,
                                padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(2, stride=2,
                                  ceil_mode=True))
        ]

        self.conv1 = nn.Sequential(OrderedDict(layer0_modules))

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = nn.Sequential(
            ConvBnRelu(encoder_outputs[4], 512),
            ConvBnRelu(512, 512),
            nn.MaxPool2d(kernel_size=2)
        )

        self.mask_clf = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                      Flatten(),
                                      nn.Linear(512, 1))

        self.up_concat5 = unetUp(in_=512, skip_=encoder_outputs[4],
                                 middle_=middles[4], out_=decoder_sizes[4])

        self.up_concat4 = unetUp(in_=decoder_sizes[4], skip_=encoder_outputs[3],
                                 middle_=middles[3], out_=decoder_sizes[3])

        self.up_concat3 = unetUp(in_=decoder_sizes[3], skip_=encoder_outputs[2],
                                 middle_=middles[2], out_=decoder_sizes[2])

        self.up_concat2 = unetUp(in_=decoder_sizes[2], skip_=encoder_outputs[1],
                                 middle_=middles[1], out_=decoder_sizes[1])

        self.up_concat1 = unetUp(in_=decoder_sizes[1], skip_=encoder_outputs[0],
                                 middle_=middles[0], out_=decoder_sizes[0])

        # hypercolumn
        self.dsv5 = UnetDsv(in_size=decoder_sizes[4], out_size=n_classes, scale_factor=16)
        self.dsv4 = UnetDsv(in_size=decoder_sizes[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv(in_size=decoder_sizes[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv(in_size=decoder_sizes[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv2d(in_channels=decoder_sizes[0], out_channels=n_classes, kernel_size=1)

        self.final = nn.Conv2d(n_classes * 5, 1, kernel_size=1, stride=1, padding=0)
        # self.final = nn.Sequential(
        #     nn.Conv2d(n_classes * 5, filter_param, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(filter_param, 1, kernel_size=1, stride=1, padding=0))

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 64 x 64 x 64
        conv2 = self.conv2(conv1)  # 256 x 64 x 64
        conv3 = self.conv3(conv2)  # 512 x 32 x 32
        conv4 = self.conv4(conv3)  # 1024 x 16 x 16
        conv5 = self.conv5(conv4)  # 2048 x 8 x 8
        center = self.center(conv5)

        up5 = self.up_concat5(center, conv5)

        up4 = self.up_concat4(up5, conv4)

        up3 = self.up_concat3(up4, conv3)

        up2 = self.up_concat2(up3, conv2)

        up1 = self.up_concat1(up2, conv1)

        dsv5 = self.dsv5(up5)
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)

        mask_bool = self.mask_clf(F.dropout2d(center, p=DROPOUT_RATE, training=self.training))
        f = F.dropout2d(torch.cat([dsv1, dsv2, dsv3, dsv4, dsv5], dim=1), p=DROPOUT_RATE, training=self.training)

        final = self.final(f)

        return final, [dsv1, dsv2, dsv3, dsv4, dsv5], mask_bool

    def freeze_encoder(self):
        freeze_params(self.conv1.parameters())
        freeze_params(self.conv2.parameters())
        freeze_params(self.conv3.parameters())
        freeze_params(self.conv4.parameters())
        freeze_params(self.conv5.parameters())

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


def freeze_params(params):
    for param in params:
        param.requires_grad = False
