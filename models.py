import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(m):
    '''
    Kaiming initialization for Conv2d layers (He et al., 2015b).
    Custom averaging init for BasicBlock projection shortcuts.
    '''
    def init_conv(module):
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight.data, nonlinearity='relu')

    def init_downsample(module):
        if isinstance(module, BasicBlock):
            ds = module.conv_downsample
            if ds:
                ds.weight.data.fill_(1 / module.in_channels)
                ds.bias.data.fill_(0)

    m.apply(init_conv)
    m.apply(init_downsample)


class ClassifierHead(nn.Module):
    def __init__(self, in_channels, out_labels):
        super().__init__()
        self.in_channels = in_channels
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_channels, out_labels)
        )

    def forward(self, x):
        return self.model.forward(x)


class BasicBlock(nn.Module):
    '''
    Two-layer basic residual block (Section 3, Figure 2 left).
    Supports shortcut options A (zero-padding) and B (1x1 projection).
    '''
    expansion = 1

    def __init__(self, in_channels, mid_channels=None, shortcut=True, downsample=False, option=None):
        super().__init__()
        assert option in {None, 'A', 'B'}, f"{option} is an invalid option"

        if mid_channels is not None:
            out_channels = mid_channels * self.expansion
        elif downsample:
            out_channels = in_channels * 2
        else:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.shortcut = shortcut
        self.option = option
        self.conv_downsample = None
        self.needs_projection = (in_channels != out_channels) or downsample

        stride = 2 if downsample else 1

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        if self.needs_projection and self.shortcut:
            assert option is not None, 'specify option A/B when downsampling'
            if self.option == 'B':
                self.conv_downsample = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        if not self.shortcut:
            res = self.model(x)
        elif not self.needs_projection:
            res = self.model(x) + x
        elif self.option == 'A':
            y = self.model(x)
            if self.downsample:
                x = F.max_pool2d(x, 1, 2)
            if self.out_channels > x.size(1):
                padding = torch.zeros(x.size(0), self.out_channels - x.size(1),
                                      x.size(2), x.size(3),
                                      device=x.device, dtype=x.dtype)
                x = torch.cat((x, padding), dim=1)
            res = y + x
        else:
            res = self.model(x) + self.conv_downsample(x)
        return F.relu(res)


class BottleneckBlock(nn.Module):
    '''
    Three-layer bottleneck block: 1x1 -> 3x3 -> 1x1 (Section 4, Figure 5).
    The 1x1 layers reduce and restore dimensions; the 3x3 layer is the bottleneck.
    Supports shortcut options A (zero-padding) and B (1x1 projection).
    '''
    expansion = 4

    def __init__(self, in_channels, mid_channels, shortcut=True, downsample=False, option=None):
        super().__init__()
        assert option in {None, 'A', 'B'}, f"{option} is an invalid option"
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = mid_channels * self.expansion
        self.downsample = downsample
        self.shortcut = shortcut
        self.option = option
        self.conv_downsample = None
        self.needs_projection = (in_channels != self.out_channels) or downsample

        stride = 2 if downsample else 1

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, self.out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )

        if self.needs_projection and self.shortcut:
            assert option is not None, 'specify option A/B for projection shortcut'
            if self.option == 'B':
                self.conv_downsample = nn.Sequential(
                    nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.out_channels)
                )

    def forward(self, x):
        if not self.shortcut:
            res = self.model(x)
        elif not self.needs_projection:
            res = self.model(x) + x
        elif self.option == 'A':
            y = self.model(x)
            if self.downsample:
                x = F.max_pool2d(x, 1, 2)
            if self.out_channels > x.size(1):
                padding = torch.zeros(x.size(0), self.out_channels - x.size(1),
                                      x.size(2), x.size(3),
                                      device=x.device, dtype=x.dtype)
                x = torch.cat((x, padding), dim=1)
            res = y + x
        else:
            res = self.model(x) + self.conv_downsample(x)
        return F.relu(res)


class CifarResNet(nn.Module):
    '''
    CIFAR-10 ResNets following Section 4.2 of He et al., 2015.
    Architecture: 6n+2 layers using 3x3 convolutions.
    Feature map sizes {32, 16, 8} with filter counts {16, 32, 64}.
    No dropout or maxout - regularization via architecture depth only.
    '''
    def __init__(self, n, residual=True, option=None):
        super().__init__()
        layers = {20, 32, 44, 56, 110}
        assert n in layers
        k = (n - 2) // 6
        modules = [
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ]
        modules += [BasicBlock(16, shortcut=residual) for _ in range(k)]

        modules.append(BasicBlock(16, shortcut=residual, downsample=True, option=option))
        modules += [BasicBlock(32, shortcut=residual) for _ in range(k - 1)]

        modules.append(BasicBlock(32, shortcut=residual, downsample=True, option=option))
        modules += [BasicBlock(64, shortcut=residual) for _ in range(k - 1)]

        modules.append(ClassifierHead(64, 10))
        self.model = nn.Sequential(*modules)
        initialize_weights(self)

    def forward(self, x):
        return self.model.forward(x)

    @staticmethod
    def transform(x):
        return x - x.mean(dim=(2, 3), keepdim=True)


class ImageNetResNet(nn.Module):
    '''
    ImageNet ResNets following Section 4.1 of He et al., 2015.
    Uses 7x7 conv + maxpool stem, then 4 stages of residual blocks.
    BasicBlock for ResNet-18/34, BottleneckBlock for ResNet-50/101/152.
    '''
    def __init__(self, n, residual=True, option=None):
        super().__init__()

        assert n in {18, 34, 50, 101, 152}, 'N must be 18, 34, 50, 101 or 152'
        if n == 18:
            layers = (2, 2, 2, 2)
            block = BasicBlock
        elif n == 34:
            layers = (3, 4, 6, 3)
            block = BasicBlock
        elif n == 50:
            layers = (3, 4, 6, 3)
            block = BottleneckBlock
        elif n == 101:
            layers = (3, 4, 23, 3)
            block = BottleneckBlock
        else:  # 152
            layers = (3, 8, 36, 3)
            block = BottleneckBlock

        self.in_channels = 64
        modules = [
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        ]

        channels_list = [64, 128, 256, 512]

        for i, num_blocks in enumerate(layers):
            channels = channels_list[i]
            for j in range(num_blocks):
                downsample = (j == 0 and i != 0)
                modules.append(block(self.in_channels, channels, downsample=downsample, shortcut=residual, option=option))
                self.in_channels = channels * block.expansion

        modules.append(ClassifierHead(self.in_channels, 1000))
        self.model = nn.Sequential(*modules)
        initialize_weights(self)

    def forward(self, x):
        return self.model.forward(x)
