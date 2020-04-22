#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:56:01 2020

@author: lds
"""

import torch
import torch.nn as nn
from .Layers import Conv2d, Flatten, LocalResponseNorm, Squeeze
from functools import partial



class GoogLeNet(nn.Module):
    def __init__(self, num_classes, mode='train'):
        super(GoogLeNet, self).__init__()
        self.num_classes = num_classes
        self.mode = mode     
        self.layers = nn.Sequential(
            Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            LocalResponseNorm(),
            Conv2d(64, 64, kernel_size=1),
            Conv2d(64, 192, kernel_size=3, padding=1),
            Conv2d(192, 64, kernel_size=1),
            Conv2d(64, 192, kernel_size=3, padding=1),
            LocalResponseNorm(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # 28x28
            Inceptionv1(192, 64, 96, 128, 16, 32, 32), # 3a
            Inceptionv1(256, 128, 128, 192, 32, 96, 64), # 3b
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Inceptionv1(480, 192, 96, 208, 16, 48, 64), # 4a
            Inceptionv1(512, 160, 112, 224, 24, 64, 64), # 4b
            Inceptionv1(512,128, 128, 256, 24, 64, 64), # 4c
            Inceptionv1(512, 112, 144, 288, 32, 64, 64), # 4d
            Inceptionv1(528, 256, 160, 320, 32, 128, 128), # 4e
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Inceptionv1(832, 256, 160, 320, 32, 128, 128), # 5a
            Inceptionv1(832, 384, 192, 384, 48, 128, 128), # 5b
            nn.AvgPool2d(7, 1),
            Squeeze(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        ) 
        if mode == 'train':
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

    def forward(self, x):   
        for idx, layer in enumerate(self.layers):
            if(idx == 13 and self.mode == 'train'):
                aux1 = self.aux1(x)
            elif(idx == 16 and self.mode == 'train'):  
                aux2 = self.aux2(x)
            x = layer(x)
        if self.mode == 'train':
            return x, aux1, aux2
        else:
            return x
    
    def init_weights(self, init_mode='VGG'):
        def init_function(m, init_mode):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_mode == 'VGG':
                    torch.nn.init.normal_(m.weight, mean=0, std=0.01)
                elif init_mode == 'XAVIER': 
                    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                    std = (2.0 / float(fan_in + fan_out)) ** 0.5
                    a = (3.0)**0.5 * std
                    with torch.no_grad():
                        m.weight.uniform_(-a, a)
                elif init_mode == 'KAMING':
                     torch.nn.init.kaiming_uniform_(m.weight)
                
                m.bias.data.fill_(0)    
        _ = self.apply(partial(init_function, init_mode=init_mode))
    
class Inceptionv1(nn.Module):
    def __init__(self, input_channel, conv1_channel, conv3_reduce_channel,
                 conv3_channel, conv5_reduce_channel, conv5_channel, pool_reduce_channel):
        super(Inceptionv1, self).__init__()
        self.conv1 = Conv2d(input_channel, conv1_channel, kernel_size=1)
        self.conv3_reduce = Conv2d(input_channel, conv3_reduce_channel, kernel_size=1)
        self.conv3 = Conv2d(conv3_reduce_channel, conv3_channel, kernel_size=3, padding=1)
        self.conv5_reduce = Conv2d(input_channel, conv5_reduce_channel, kernel_size=1)
        self.conv5 = Conv2d(conv5_reduce_channel, conv5_channel, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_reduce = Conv2d(input_channel, pool_reduce_channel, kernel_size=1)
    
    def forward(self, x):
        output_conv1 = self.conv1(x)
        output_conv3 = self.conv3(self.conv3_reduce(x))
        output_conv5 = self.conv5(self.conv5_reduce(x))
        output_pool = self.pool_reduce(self.pool(x))
        outputs = torch.cat([output_conv1, output_conv3, output_conv5, output_pool], dim=1)
        return outputs  

class InceptionAux(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(InceptionAux, self).__init__()
        self.layers = nn.Sequential(
            nn.AvgPool2d(5, 3),
            Conv2d(input_channel, 128, 1),
            Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
            )
    
    def forward(self, x):
        x = self.layers(x)
        return x

if __name__ == '__main__':
    net = GoogLeNet(1000)
