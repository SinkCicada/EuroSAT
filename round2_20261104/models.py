import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet
from models_base import SE, CBAM, SEBlock, CBAMBlock, complexity


def make_model(variant, classes=10):
    # Build the common backbone before extra modules: matching initial backbone
    # weights across B/C/D at the same seed, unlike independent constructions.
    model=ResNet(BasicBlock,[2,2,2,2],num_classes=classes)
    if variant=='original':return model
    if variant not in ['small','small_se','small_cbam']:raise ValueError(variant)
    model.conv1=nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    nn.init.kaiming_normal_(model.conv1.weight,mode='fan_out',nonlinearity='relu')
    # Keep maxpool. layer3=8x8 and layer4=4x4 for a 64x64 image.
    if variant in ['small_se','small_cbam']:
        cls,module=(SEBlock,SE) if variant=='small_se' else (CBAMBlock,CBAM)
        for block in model.layer3:
            block.__class__=cls
            block.attention=module(block.conv2.out_channels)
            for m in block.attention.modules():
                if isinstance(m,nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
    return model
