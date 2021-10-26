import torch
from models.network.UNet import UNet
from models.network.R2Unet import R2U_Net
from models.network.AttUNet import AttU_Net
from models.network.R2AttUNet import R2AttU_Net

def segmentation_model(in_channels = 3, n_classes = 1, seg_model_name = 'UNet'):
    if seg_model_name == 'UNet':
        seg_model = UNet(in_channels, n_classes)
    elif seg_model_name == 'R2UNet':
        seg_model = R2U_Net(in_channels, n_classes)
    elif seg_model_name == 'AttUNet':
        seg_model = AttU_Net(in_channels, n_classes)
    elif seg_model_name == 'R2AttUNet':
        seg_model = R2AttU_Net(in_channels, n_classes)

    return seg_model 