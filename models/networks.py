# encoding: utf-8
import copy
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import random
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from torch import nn, optim
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.models.resnet import Bottleneck, resnet50
from torchvision.transforms import functional

# from .resnet import ResNet

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0.0)


class PBN_modify(nn.Module):
    def __init__(self, input_dim, num_classes, num_reduction=512):
        super(PBN_modify, self).__init__()
        self.input_dim = input_dim
        self.num_reduction = num_reduction

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))
        
        reduction2 = []
        reduction2 += [nn.Conv2d(input_dim, num_reduction, kernel_size=1, bias=True)]
        reduction2 += [nn.BatchNorm2d(num_reduction)]
        reduction2 += [nn.LeakyReLU(0.1)]
        reduction2 = nn.Sequential(*reduction2)
        reduction2.apply(weights_init_kaiming)
        self.reduction2 = reduction2        

        # self.bn1 = nn.BatchNorm1d(self.num_reduction)
        self.bn2 = nn.BatchNorm1d(self.num_reduction)

        # self.classifier1 = nn.Linear(self.num_reduction, num_classes)
        # self.classifier1.apply(weights_init_classifier)

        self.classifier2 = nn.Linear(self.num_reduction, num_classes)
        self.classifier2.apply(weights_init_classifier)
        # self.classifier = classifier

    def forward(self, x):
        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)
        x2 = x1 + x2
        # ------ for x2 -------
        x2_1 = self.reduction2(x2) 
        # print(x2_1.shape)
        # x2_2 = torch.squeeze(x2_1) # 跟这里有关,看一下x2_1的维度，注意N可能为1
        x2_2 = torch.squeeze(x2_1, 3)
        x2_2 = torch.squeeze(x2_2, 2)
        x2_3 = self.bn2(x2_2)
        x2_4 = self.classifier2(x2_3)        

        return x2_2, x2_3, x2_4  # x5 for triplet; x6 for inference; x7 for softmax

class Conv_BN(nn.Module):
    def __init__(self, input_dim, num_classes, num_reduction=512):
        super(Conv_BN, self).__init__()
        self.input_dim = input_dim
        self.num_reduction = num_reduction

        reduction2 = []
        reduction2 += [nn.Conv2d(input_dim, num_reduction, kernel_size=1, bias=True)]
        reduction2 += [nn.BatchNorm2d(num_reduction)]
        reduction2 += [nn.LeakyReLU(0.1)]
        reduction2 = nn.Sequential(*reduction2)
        reduction2.apply(weights_init_kaiming)
        self.reduction2 = reduction2        

        self.bn2 = nn.BatchNorm1d(self.num_reduction)

        self.classifier2 = nn.Linear(self.num_reduction, num_classes)
        self.classifier2.apply(weights_init_classifier)

    def forward(self, x):
        x = torch.unsqueeze(x, 2)
        x = torch.unsqueeze(x, 3)
        x2_1 = self.reduction2(x) 
        # print(x2_1.shape)
        x2_2 = torch.squeeze(x2_1, 3) # notice N == 1
        x2_2 = torch.squeeze(x2_2, 2)
        x2_3 = self.bn2(x2_2)
        x2_4 = self.classifier2(x2_3)        

        return x2_2, x2_3, x2_4  

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

class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
    
    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h-rh)
            sy = random.randint(0, w-rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx+rh, sy:sy+rw] = 0
            x = x * mask
        return x

class BatchCrop(nn.Module):
    def __init__(self, ratio):
        super(BatchCrop, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rw = int(self.ratio * w)
            start = random.randint(0, h-1)
            if start + rw > h:
                select = list(range(0, start+rw-h)) + list(range(start, h))
            else:
                select = list(range(start, start+rw))
            mask = x.new_zeros(x.size())
            mask[:, :, select, :] = 1
            x = x * mask
        return x

class ResNetBuilder(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes=None, last_stride=1, pretrained=False):
        super().__init__()
        self.base = ResNet(last_stride)
        if pretrained:
            model_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            self.base.load_param(model_zoo.load_url(model_url))

        self.num_classes = num_classes
        if num_classes is not None:
            self.bottleneck = nn.Sequential(
                nn.Linear(self.in_planes, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5)
            )
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier = nn.Linear(512, self.num_classes)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        global_feat = self.base(x)
        global_feat = F.avg_pool2d(global_feat, global_feat.shape[2:])  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        if self.training and self.num_classes is not None:
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)
            return [global_feat], [cls_score]
        else:
            return global_feat

    def get_optim_policy(self):
        base_param_group = self.base.parameters()
        if self.num_classes is not None:
            add_param_group = itertools.chain(self.bottleneck.parameters(), self.classifier.parameters())
            return [
                {'params': base_param_group},
                {'params': add_param_group}
            ]
        else:
            return [
                {'params': base_param_group}
            ]

class BFE(nn.Module):
    def __init__(self, num_classes, width_ratio=0.5, height_ratio=0.5):
        super(BFE, self).__init__()
        resnet = resnet50(pretrained=True)
        resnet.fc = nn.Sequential()
        self.model = resnet
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

        sub_resnet = resnet50(pretrained=True)
        sub_resnet.fc = nn.Sequential()
        self.sub_model = sub_resnet
        self.sub_model.layer4[0].downsample[0].stride = (1,1)
        self.sub_model.layer4[0].conv2.stride = (1,1)

        # global branch
        self.bottleneck_g1 = Bottleneck(2048, 512)
        self.bottleneck_g1_1 = Bottleneck(1024, 256)
        self.bottleneck_g2 = Bottleneck(2048, 512)
        self.bottleneck_g2_1 = Bottleneck(1024, 256)
        self.PBN1 = PBN_modify(2048, num_classes, num_reduction=512) # 到这来
        self.PBN1_1 = PBN_modify(1024, num_classes, num_reduction=256) # 到这来

        self.PBN2 = PBN_modify(2048, num_classes, num_reduction=512)
        self.PBN2_1 = PBN_modify(1024, num_classes, num_reduction=256)
        self.convbn = Conv_BN(1536, num_classes, num_reduction=512)

    def forward_once(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x1 = self.model.layer3(x)
        x = self.model.layer4(x1) 
        return x, x1

    def forward_twice(self, x):
        x = self.sub_model.conv1(x)
        x = self.sub_model.bn1(x)
        x = self.sub_model.relu(x)
        x = self.sub_model.maxpool(x)

        x = self.sub_model.layer1(x)
        x = self.sub_model.layer2(x)
        x3 = self.sub_model.layer3(x)
        x = self.sub_model.layer4(x3) 
        return x, x3

    def forward(self, input1, input2):
        '''
        首先得到两个网络的框架（是否共享参数，有待考虑）
        然后将两个网络的结果进行合并处理
        接着经过一个PBN即可，标签是共享的（一一对应的）
        '''
        # handle stream1
        feature_map1, feature_map1_1 = self.forward_once(input1)
        feature1 = self.bottleneck_g1(feature_map1)
        x1_1, x1_2, x1_3 = self.PBN1(feature1)

        feature1_1 = self.bottleneck_g1_1(feature_map1_1)
        x1_4, x1_5, x1_6 = self.PBN1_1(feature1_1)  

        # handle stream2
        feature_map2, feature_map2_1 = self.forward_twice(input2)
        feature2 = self.bottleneck_g2(feature_map2)
        x2_1, x2_2, x2_3 = self.PBN2(feature2)

        feature2_1 = self.bottleneck_g2_1(feature_map2_1)
        x2_4, x2_5, x2_6 = self.PBN2_1(feature2_1)

        # concatenate stream1 and stream2
        con_feature = torch.cat((x1_2, x1_5, x2_2, x2_5), 1)
        x3_1, x3_2, x3_3 = self.convbn(con_feature)
        
        # --- global 分支 --
        predict = []
        triplet_features = []
        softmax_features = []

       # add stream1 feature
        softmax_features.append(x1_3)
        triplet_features.append(x1_1)
        predict.append(x1_2)

        softmax_features.append(x1_6)
        triplet_features.append(x1_4)
        predict.append(x1_5)

        # add stream2 feature
        softmax_features.append(x2_3)
        triplet_features.append(x2_1)
        predict.append(x2_2) 

        softmax_features.append(x2_6)
        triplet_features.append(x2_4)
        predict.append(x2_5) 

        # concate feature
        softmax_features.append(x3_3)
        triplet_features.append(x3_1)
        predict.append(x3_2)       

        if self.training:
            return triplet_features, softmax_features
        else:
            return torch.cat(predict, 1)

class Resnet(nn.Module):
    def __init__(self, num_classes, resnet=None):
        super(Resnet, self).__init__()
        if not resnet:
            resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
            resnet.layer4
        )
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Linear(2048, num_classes)

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)

        x = self.global_avgpool(x).squeeze()
        feature = self.softmax(x)
        if self.training:
            return [], [feature]
        else:
            return feature

    def get_optim_policy(self):
        return self.parameters()

class IDE(nn.Module):
    def __init__(self, num_classes, resnet=None):
        super(IDE, self).__init__()
        if not resnet:
            resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
            resnet.layer4
        )
        self.global_avgpool = nn.AvgPool2d(kernel_size=(12, 4))

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)

        feature = self.global_avgpool(x).squeeze()
        if self.training:
            return [feature], []
        else:
            return feature

    def get_optim_policy(self):
        return self.parameters()


# bfe = BFE(512)
# print(bfe)