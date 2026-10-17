import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import timm
from .aspp import build_aspp

class seg_model(nn.Module):
    def __init__(self, num_class):
        super(seg_model, self).__init__()
        self.num_class = num_class
        self.encoder = timm.create_model('resnet18', pretrained= True, features_only=True, out_indices=[1,4])
    
        low_level_inplanes = 512 #for resnet18
        aspp_outplanes = 256
        self.aspp = build_aspp(inplanes = low_level_inplanes, outplanes = aspp_outplanes)
        
        #decoder
        self.conv1 = nn.Conv2d(64, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, self.num_class, kernel_size=1, stride=1))
        
            

 
    def forward(self, inp, require_feature=False):

        features = self.encoder(inp)
        low_level_feat, x = features
        x = self.aspp(x)
        
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        concat_features = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(concat_features)
        output = F.interpolate(x, size=inp.size()[2:], mode='bilinear', align_corners=True)

        if require_feature:
            return output, features
        else:
            return output

