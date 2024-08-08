import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

def conv_b(in_dim, out_dim):
    block = nn.Sequential(
        #Jeehyun : stride and padding to fit the Cityscapes. BatchNorm is from reference
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1,padding=1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        #nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )
    return block

def maxpool():
    block = nn.Sequential(
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )
    return block


def bridge(in_dim, out_dim):
    block = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )
    return block


def conv_trans(in_dim, out_dim):

    block = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
    )
    return block

def output_b(in_dim, out_dim):
    block = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0), #TODO(jeehyun) should check 1*1 conv

    )
    return block


class UNet(nn.Module):

    def __init__(self, in_dim, num_filters, num_classes):
        super(UNet, self).__init__()
        self.in_dim = in_dim
        #self.out_dim = out_dim
        self.num_filters = num_filters
        self.num_classes = num_classes

        self.maxpool = maxpool()
        self.down_1 = conv_b(self.in_dim, self.num_filters)
        self.down_2 = conv_b(self.num_filters*1, self.num_filters*2)
        self.down_3 = conv_b(self.num_filters*2, self.num_filters*4)
        self.down_4 = conv_b(self.num_filters*4, self.num_filters*8)
        self.bridge = bridge(self.num_filters*8, self.num_filters*16)
        self.trans_1 = conv_trans(self.num_filters*16, self.num_filters*8)
        self.up_1 = conv_b(self.num_filters*16, self.num_filters*8)
        self.trans_2 = conv_trans(self.num_filters*8, self.num_filters*4)
        self.up_2 = conv_b(self.num_filters*8, self.num_filters*4)
        self.trans_3 = conv_trans(self.num_filters*4, self.num_filters*2)
        self.up_3 = conv_b(self.num_filters*4, self.num_filters*2)
        self.trans_4 = conv_trans(self.num_filters*2, self.num_filters*1)
        self.up_4 = conv_b(self.num_filters*2, self.num_filters*1)
        self.final = output_b(self.num_filters*1, self.num_classes)

    def forward(self, x):
        d1 = self.down_1(x)
        tmp_d1 = self.maxpool(d1)
        d2 = self.down_2(tmp_d1)
        tmp_d2 = self.maxpool(d2)
        d3 = self.down_3(tmp_d2)
        tmp_d3 = self.maxpool(d3)
        d4 = self.down_4(tmp_d3)
        tmp_d4 = self.maxpool(d4)
        bridge = self.bridge(tmp_d4)

        t1 = self.trans_1(bridge)
        cc1 = torch.cat([t1, d4], dim=1)
        u1 = self.up_1(cc1)
        t2 = self.trans_2(u1)
        cc2 = torch.cat([t2, d3], dim=1)
        u2 = self.up_2(cc2)
        t3 = self.trans_3(u2)
        cc3 = torch.cat([t3, d2], dim=1)
        u3 = self.up_3(cc3)
        t4 = self.trans_4(u3)
        cc4 = torch.cat([t4, d1], dim=1)
        u4 = self.up_4(cc4)
        out = self.final(u4)
        return out






