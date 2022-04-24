import torch
from torch import nn
import torch.nn.functional as F


def load_network(device, path=None, **resnet_params):
    in_channels = resnet_params['in_channels']
    base_filters = resnet_params['base_filters']
    kernel_size = resnet_params['kernel_size']
    stride = resnet_params['stride']
    groups = resnet_params['groups']
    n_block = resnet_params['n_block']
    n_classes = resnet_params['n_classes']
    downsample_gap = resnet_params['downsample_gap']
    increasefilter_gap = resnet_params['increasefilter_gap']
    num_features = resnet_params['num_features']

    # initial a residual network
    model = ResNet(in_channels=in_channels,
                   base_filters=base_filters,
                   kernel_size=kernel_size,
                   stride=stride,
                   groups=groups,
                   n_block=n_block,
                   n_classes=n_classes,
                   downsample_gap=downsample_gap,
                   increasefilter_gap=increasefilter_gap,
                   num_features=num_features)
    model = model.to(device)

    # if exists, load the pre-trained model
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
    return model


def test_forward_pass():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resnet_params = dict()
    resnet_params['num_features'] = 64
    resnet_params['in_channels'] = 1
    resnet_params['base_filters'] = 64
    resnet_params['kernel_size'] = 16
    resnet_params['stride'] = 2
    resnet_params['groups'] = 32
    resnet_params['n_block'] = 48
    resnet_params['n_classes'] = 4
    resnet_params['downsample_gap'] = 6
    resnet_params['increasefilter_gap'] = 12

    model = load_network(device, **resnet_params)

    signal_length = 6000

    no_channels = 1

    batch_size = 3
    example_source = torch.rand((batch_size, no_channels, signal_length)).to(device)

    features = model.get_features(example_source)
    print(features.size())

    result = model(example_source)
    print(result.size())


def run():
    test_forward_pass()


class Conv1dPadSame(nn.Module):
    """
    Extend nn.Conv1d to support same padding
    Code from: https://github.com/hsd1503/resnet1d/blob/master/resnet1d.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(Conv1dPadSame, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = nn.Conv1d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              groups=self.groups)

    def forward(self, x):

        # compute padding shape
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        pad = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = pad // 2
        pad_right = pad - pad_left

        output = F.pad(x, (pad_left, pad_right), mode="constant", value=0)
        output = self.conv(output)

        return output


class MaxPool1dPadSame(nn.Module):
    """
    Extend nn.MaxPool1d to support same padding
    Code from: https://github.com/hsd1503/resnet1d/blob/master/resnet1d.py
    """
    def __init__(self, kernel_size):
        super(MaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):

        # compute pad shape
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        pad = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = pad // 2
        pad_right = pad - pad_left

        output = F.pad(x, (pad_left, pad_right), mode="constant", value=0)
        output = self.max_pool(output)

        return output


class ResidualBlock(nn.Module):
    """
    First residual block: Conv -> BN -> ReLU -> Dropout -> Conv
    Rest residual block:  BN -> ReLU -> Dropout -> Conv -> BN -> ReLU -> Dropout -> Conv
    Code from: https://github.com/hsd1503/resnet1d/blob/master/resnet1d.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, downsample, stride, groups, is_first_block=False):
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.downsample = downsample  # bool
        self.stride = stride if downsample else 1

        self.groups = groups

        self.is_first_block = is_first_block

        # first convolutional layer
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.conv1 = Conv1dPadSame(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   groups=self.groups)

        # second convolutional layer
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.conv2 = Conv1dPadSame(in_channels=self.out_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   stride=1,
                                   groups=self.groups)

        self.max_pool = MaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):

        shortcut = x
        output = x

        if not self.is_first_block:
            output = self.bn1(output)
            output = self.relu1(output)
            output = self.dropout1(output)
        output = self.conv1(output)

        output = self.bn2(output)
        output = self.relu2(output)
        output = self.dropout2(output)
        output = self.conv2(output)

        # if downsample, also downsample shortcut
        if self.downsample:
            shortcut = self.max_pool(shortcut)

        # if expand channel, also pad zeros to shortcut
        if self.out_channels != self.in_channels:
            shortcut = shortcut.transpose(-1, -2)
            channel_1 = (self.out_channels - self.in_channels) // 2
            channel_2 = self.out_channels - self.in_channels - channel_1
            shortcut = F.pad(shortcut, (channel_1, channel_2), mode="constant", value=0)
            shortcut = shortcut.transpose(-1, -2)

        output += shortcut

        return output


class ResNet(nn.Module):
    """
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        output: (n_samples, n_classes)

    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes

    Code from: https://github.com/hsd1503/resnet1d/blob/master/resnet1d.py (modify the last several layers)
    """
    def __init__(self, in_channels, base_filters, kernel_size, stride, groups,
                 n_block, n_classes=4, downsample_gap=2, increasefilter_gap=4, num_features=32, verbose=False):
        super(ResNet, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.n_block = n_block
        self.num_features = num_features
        self.verbose = verbose

        self.downsample_gap = downsample_gap  # 2 for base model
        self.increasefilter_gap = increasefilter_gap  # 4 for base model

        # first block
        self.first_block_conv = Conv1dPadSame(in_channels=in_channels,
                                              out_channels=base_filters,
                                              kernel_size=kernel_size,
                                              stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()

        out_channels = base_filters

        # residual blocks
        self.residualblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            is_first_block = True if i_block == 0 else False

            # downsample at every <self.downsample_gap> blocks
            downsample = True if i_block % self.downsample_gap == 1 else False

            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every <self.increasefilter_gap> blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            res_block = ResidualBlock(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      groups=self.groups,
                                      downsample=downsample,
                                      is_first_block=is_first_block)
            self.residualblock_list.append(res_block)

        # feature extraction
        self.feature_bn = nn.BatchNorm1d(out_channels)
        self.feature_relu = nn.ReLU()
        self.feature_dropout = nn.Dropout(p=0.5)
        self.feature_layer = nn.Linear(out_channels, self.num_features)
        self.feature_sigmoid = nn.Sigmoid()

        # final prediction
        self.final_dropout = nn.Dropout(p=0.5)
        self.final_dense = nn.Linear(self.num_features, n_classes)
        self.final_softmax = nn.Softmax(dim=1)

    def get_features(self, x):

        output = x

        # first block
        if self.verbose:
            print("Input shape:", output.shape)
        output = self.first_block_conv(output)
        if self.verbose:
            print("After first conv:", output.shape)
        output = self.first_block_bn(output)
        output = self.first_block_relu(output)

        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            res_block = self.residualblock_list[i_block]
            if self.verbose:
                print("i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}".format(
                    i_block, res_block.in_channels, res_block.out_channels, res_block.downsample))
            output = res_block(output)
            if self.verbose:
                print(output.shape)

        # feature extraction
        output = self.feature_bn(output)
        output = self.feature_relu(output)
        output = output.mean(-1)
        if self.verbose:
            print('Final pooling', output.shape)
        output = self.feature_dropout(output)
        output = self.feature_sigmoid(self.feature_layer(output))
        if self.verbose:
            print('Feature shape', output.shape)

        return output

    def forward(self, x):

        # feature extraction
        output = self.get_features(x)

        # final prediction
        output = self.final_dropout(output)
        output = self.final_dense(output)
        if self.verbose:
            print("Dense: ", output.shape)
        output = self.final_softmax(output)
        if self.verbose:
            print('Softmax', output.shape)

        return output


if __name__ == "__main__":
    run()


