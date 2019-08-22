# weight decay = 0.0001 momentum = 0.9 in weight initialization nad BN

import torch.nn as nn


# CIFAR-10 dataset
class ResNet(nn.Module):
    def __init__(self, block_numbers, class_numbers):
        super(ResNet, self).__init__()
        # input layer
        self.input_conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)  # input_size 유지
        self.input_batach = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU(inplace=False)

        # 32block, output_size = 32x32
        self.layer_32 = self.make_layers(block_numbers=block_numbers, in_channels=16, out_channels=16)
        # 16block output_size = 16x16
        self.layer_16 = self.make_layers(block_numbers=block_numbers, in_channels=16, out_channels=32,
                                         down_sampling=True)
        # 8block output_size = 8x8
        self.layer_8 = self.make_layers(block_numbers=block_numbers, in_channels=32, out_channels=64,
                                        down_sampling=True)

        # last_layer , output_size = 1x1로 조정
        self.last_pool = nn.AvgPool2d(kernel_size=8)
        self.last_linear = nn.Linear(in_features=64,
                                     out_features=class_numbers)  # 인풋 = (64, 1, 1).Flatten(), 아웃풋 = 클래스 개수

    def forward(self, x):
        # input_layer
        y = self.input_conv(x)
        y = self.input_batach(y)
        y = self.relu(y)

        # 레이어 쌓기
        y = self.layer_32(y)
        y = self.layer_16(y)
        y = self.layer_8(y)

        # last layer
        print(y.shape)
        y = self.last_pool(y)
        print(y.shape)
        y = y.view(y.size(0), -1) # 이게 없으면 사이즈에러
        print(y.shape)
        y = self.last_linear(y)

        return y

    def make_layers(self, block_numbers, in_channels, out_channels, down_sampling=False):

        if down_sampling:
            # 다운샘플링되는 경우 16 -> 32, 32 -> 64 / layer_16, layer_8
            block_list = nn.ModuleList([BasicBlock(in_channels=in_channels, out_channels=out_channels,
                                                   stride=2, down_sampling=down_sampling)])
        else:
            # 다운샘플링하지 않는 경우는 layer_32만
            block_list = nn.ModuleList([BasicBlock(in_channels=in_channels, out_channels=out_channels,
                                                   stride=1)])

        for i in range(block_numbers - 1):
            # 이미 다운샘플링이 됐으므로 in_channel과 out_channel은 out_channel값으로
            block_list.append(BasicBlock(in_channels=out_channels, out_channels=out_channels,
                                         stride=1))

        return nn.Sequential(*block_list)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, down_sampling=False):
        super(BasicBlock, self).__init__()

        self.down_sample = down_sampling
        # layer1
        # 스트라이드는 인풋 stride를 이용 = 다운샘플링이 일어나는 경우
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)  # input_size 유지
        self.batch1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        # layer2
        # 다운샘플링이 일어나던 말던 두번째 레이어는 무조건 stride=1,
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.batch2 = nn.BatchNorm2d(num_features=out_channels)

        # 다운샘플링이 일어난경우 identity mapping 사용
        if self.down_sample:
            self.identity_mapping = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.identity_mapping = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.batch1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.batch2(y)

        # if self.down_sample:
        if self.identity_mapping is not None:
            x = self.identity_mapping(x)

        y = y + x
        y = self.relu(y)

        return y

