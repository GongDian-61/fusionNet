import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3)
        self.norm1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        # x = F.pad(inputs, (3, 3, 3, 3), mode='constant', value=0) # necessary?
        x1 = F.relu(self.norm1(self.conv1(x)))
        #x11 = F.relu(self.conv1(x))
        x2 = F.relu(self.norm2(self.conv2(F.max_pool2d(x1, kernel_size=4, stride=2,padding=1))))
        x22 = F.relu(self.conv2(F.max_pool2d(x1,kernel_size=4,stride=2)))
        x3 = F.relu(self.norm3(self.conv3(F.max_pool2d(x2, kernel_size=4, stride=2,padding=1))))
        x4 = F.relu(self.norm4(self.conv4(F.max_pool2d(x3, kernel_size=4, stride=2,padding=1))))
        return x1, x2, x3, x4


class Decoder(nn.Module):
    def __init__(self,out_channels):
        super(Decoder, self).__init__()
        self.upconv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=3)
    
    def forward(self, x4, x3, x2, x1):
        x = F.relu(self.upconv1(F.interpolate(x4, scale_factor=2, mode='nearest')))
        x = F.interpolate(x,size=x3.shape[-2:], mode='bilinear', align_corners=False)
        x = F.relu(self.upconv2(F.interpolate(x + x3, scale_factor=2, mode='nearest')))
        x = F.relu(self.upconv3(F.interpolate(x + x2, scale_factor=2, mode='nearest')))
        x = torch.sigmoid(self.final_conv(x+x1))
        return x

class EncoderDecoder(nn.Module):
    def __init__(self, in_channels=1):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder()

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        output = self.decoder(x4, x3, x2, x1)
        return output
