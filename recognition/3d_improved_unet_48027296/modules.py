"""
    File name: modules.py
    Author: Louisa Wu
    Date created: 13/10/2025
    Date last modified: 23/10/2025
    Python Version: 3.9.23
    Description: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#### 2D UNet ####

class UNet2D(nn.Module):
    """
    2D UNet architecture from: 
    O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image 
    Segmentation,” in Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015, 
    ser. Lecture Notes in Computer Science, N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, 
    Eds. Cham: Springer International Publishing, 2015, pp. 234-241.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        # encoder 
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # bottleneck 
        self.bottleneck = conv_block(512, 1024)

        # decoder 
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        # final output layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # encoder 
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # bottleneck 
        b = self.bottleneck(self.pool4(e4))

        # decoder with skip connections 
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))
    

#### 3D UNet ####

class UNet3D(nn.Module):
    """ 
    3D UNet architecture from: 
    O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, “3D U-Net: Learning 
    Dense Volumetric Segmentation from Sparse Annotation,” in Medical Image Computing and Computer-
    Assisted Intervention - MICCAI 2016, ser. Lecture Notes in Computer Science, S. Ourselin, L. 
    Joskowicz, M. R. Sabuncu, G. Unal, and W. Wells, Eds. Cham: Springer International Publishing, 
    2016, pp. 424-432.
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super().__init__()
        features = init_features

        # encoder 
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool3d(2)

        # bottleneck 
        self.bottleneck = self._block(features * 8, features * 16)

        # decoder 
        self.up4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block(features * 16, features * 8)
        self.up3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(features * 8, features * 4)
        self.up2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2)
        self.up1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)

        # final output 
        self.final = nn.Conv3d(features, out_channels, kernel_size=1)

    def _block(self, in_channels, out_channels):
        """ 
        Each block: 2 * (3D Convolution --> Batch Normalisation --> ReLU activation function)
        """
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # encoder 
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # decoder with skip connections
        dec4 = self.up4(bottleneck)
        dec4 = self.decoder4(torch.cat((dec4, enc4), dim=1))
        dec3 = self.up3(dec4)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))
        dec2 = self.up2(dec3)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))

        # final segmentation map 
        return torch.sigmoid(self.final(dec1))
    

#### Improved 3D UNet ####

class ContextModule(nn.Module):
    """Pre-activation residual block with dropout"""
    def __init__(self, in_channels, out_channels, dropout_p=0.3):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(in_channels)
        self.lrelu1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout3d(p=dropout_p)
        
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.lrelu2 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Residual connection (1x1 conv if channels change)
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        
        out = self.norm1(x)
        out = self.lrelu1(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.lrelu2(out)
        out = self.conv2(out)
        
        return out + residual


class LocalizationModule(nn.Module):
    """3x3x3 conv followed by 1x1x1 conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        return out


class Improved3DUNet(nn.Module):
    """
    Improved 3D UNet with deep supervision.

    Improved 3D UNet architecture from: 
    F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor 
    Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” 
    Feb. 2018. [Online]. Available: https://arxiv.org/abs/1802.10508v1. 
    
    Args:
        in_channels: Number of input channels (default: 1)
        num_classes: Number of output classes (default: 6)
        base_filters: Base number of filters (default: 16)
        dropout_p: Dropout probability (default: 0.3)
    """
    def __init__(self, in_channels=1, num_classes=6, base_filters=16, dropout_p=0.3):
        super().__init__()
        f = base_filters
        
        # Context pathway (encoder)
        self.initial_conv = nn.Conv3d(in_channels, f, kernel_size=3, padding=1)
        self.context1 = ContextModule(f, f, dropout_p)

        self.down1 = nn.Conv3d(f, f, kernel_size=3, stride=2, padding=1)
        self.context2 = ContextModule(f, f * 2, dropout_p)

        self.down2 = nn.Conv3d(f * 2, f * 2, kernel_size=3, stride=2, padding=1)
        self.context3 = ContextModule(f * 2, f * 4, dropout_p)

        self.down3 = nn.Conv3d(f * 4, f * 4, kernel_size=3, stride=2, padding=1)
        self.context4 = ContextModule(f * 4, f * 8, dropout_p)

        self.down4 = nn.Conv3d(f * 8, f * 8, kernel_size=3, stride=2, padding=1)
        self.context5 = ContextModule(f * 8, f * 16, dropout_p)
        
        # Localization pathway (decoder)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(f * 16, f * 8, kernel_size=3, padding=1)
        )
        self.loc4 = LocalizationModule(f * 16, f * 8)
        self.seg4 = nn.Conv3d(f * 8, num_classes, kernel_size=1)
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(f * 8, f * 4, kernel_size=3, padding=1)
        )
        self.loc3 = LocalizationModule(f * 8, f * 4)
        self.seg3 = nn.Conv3d(f * 4, num_classes, kernel_size=1)
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(f * 4, f * 2, kernel_size=3, padding=1)
        )
        self.loc2 = LocalizationModule(f * 4, f * 2)
        self.seg2 = nn.Conv3d(f * 2, num_classes, kernel_size=1)
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(f * 2, f, kernel_size=3, padding=1)
        )
        self.loc1 = LocalizationModule(f * 2, f)
        self.final = nn.Conv3d(f, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Context pathway (encoder)
        c1 = self.context1(self.initial_conv(x))

        d1 = self.down1(c1)
        c2 = self.context2(d1)

        d2 = self.down2(c2)
        c3 = self.context3(d2)

        d3 = self.down3(c3)
        c4 = self.context4(d3)

        d4 = self.down4(c4)
        c5 = self.context5(d4)
        
        # Localization pathway (decoder) with deep supervision
        u4 = self.up4(c5)
        l4 = self.loc4(torch.cat([u4, c4], dim=1))
        s4 = self.seg4(l4)
        
        u3 = self.up3(l4)
        l3 = self.loc3(torch.cat([u3, c3], dim=1))
        s3 = self.seg3(l3)
        
        u2 = self.up2(l3)
        l2 = self.loc2(torch.cat([u2, c2], dim=1))
        s2 = self.seg2(l2)
        
        u1 = self.up1(l2)
        l1 = self.loc1(torch.cat([u1, c1], dim=1))
        out = self.final(l1)
        
        # element-wise sum of all segmentation outputs
        out = out + F.interpolate(s2, size=out.shape[2:], mode='trilinear', align_corners=False)
        out = out + F.interpolate(s3, size=out.shape[2:], mode='trilinear', align_corners=False)
        out = out + F.interpolate(s4, size=out.shape[2:], mode='trilinear', align_corners=False)
        
        return out
    