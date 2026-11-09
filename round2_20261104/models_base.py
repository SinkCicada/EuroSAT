"""Classic ResNet-18 and attention in each residual branch before addition."""
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet


class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(),
                                nn.Conv2d(hidden, channels, 1, bias=False))

    def forward(self, x):
        return x * self.fc(x.mean((2, 3), keepdim=True)).sigmoid()


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(),
                                nn.Conv2d(hidden, channels, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        channel = (self.fc(x.mean((2, 3), keepdim=True)) +
                   self.fc(x.amax((2, 3), keepdim=True))).sigmoid()
        x = x * channel
        spatial = self.spatial(torch.cat([x.mean(1, keepdim=True),
                                         x.amax(1, keepdim=True)], dim=1)).sigmoid()
        return x * spatial


class AttentionBlock(BasicBlock):
    attention_class = nn.Identity

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = self.attention_class(self.conv2.out_channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.attention(self.bn2(self.conv2(out)))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class SEBlock(AttentionBlock):
    attention_class = SE


class CBAMBlock(AttentionBlock):
    attention_class = CBAM


def make_model(variant, classes=10):
    block = {'baseline': BasicBlock, 'se': SEBlock, 'cbam': CBAMBlock}[variant]
    # Original 7x7 stride-2 stem and max pool retained; only final head changes.
    return ResNet(block, [2, 2, 2, 2], num_classes=classes)


def complexity(model, size=64):
    """Conv2d + Linear MACs only; explicitly excludes pooling/gates/normalization."""
    total = 0
    handles = []
    def hook(module, inputs, output):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            total += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        else:
            total += output.numel() * module.in_features
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size, device=next(model.parameters()).device))
    for handle in handles:
        handle.remove()
    return {'parameters': sum(p.numel() for p in model.parameters()), 'conv_linear_macs': total,
            'input_shape': [1, 3, size, size], 'macs_definition': 'Conv2d+Linear only; one multiply-accumulate = one MAC'}
