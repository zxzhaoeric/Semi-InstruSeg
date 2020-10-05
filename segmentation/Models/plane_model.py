import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from .block import *
import torch.nn.functional as F

class Conv2dReLU(nn.Module):
    """
    [Conv2d(in_channels, out_channels, kernel),
    BatchNorm2d(out_channels),
    ReLU,]
    """
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, bn=False):
        super(Conv2dReLU, self).__init__()
        modules = OrderedDict()
        modules['conv'] = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
        if bn:
            modules['bn'] = nn.BatchNorm2d(out_channels)
        modules['relu'] = nn.ReLU(inplace=True)
        self.l = nn.Sequential(modules)

    def forward(self, x):
        x = self.l(x)
        return x


class UNetModule(nn.Module):
    """

    [Conv2dReLU(in_channels, out_channels, 3),
    Conv2dReLU(out_channels, out_channels, 3)]
    """
    def __init__(self, in_channels, out_channels, padding=1, bn=False):
        super(UNetModule, self).__init__()
        self.l = nn.Sequential(OrderedDict([
            ('conv1', Conv2dReLU(in_channels, out_channels, 3, padding=padding, bn=bn)),
            ('conv2', Conv2dReLU(out_channels, out_channels, 3, padding=padding, bn=bn))
            ]))

    def forward(self, x):
        x = self.l(x)
        return x

class Interpolate(nn.Module):
    """
    Wrapper function of interpolate/UpSample Module
    """
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False):
        super(Interpolate, self).__init__()
        self.fn = lambda x: nn.functional.interpolate(x, scale_factor=scale_factor,
            mode=mode, align_corners=align_corners)

    def forward(self, x):
        return self.fn(x)


class UNet(nn.Module):
    """
    UNet implementation
    pretrained model: None
    Note: this implementation doesn't strictly follow UNet, the kernel sizes are halfed
    """
    def __init__(self, in_channels, num_classes, bn=False):
        super(UNet, self).__init__()

        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample = Interpolate(scale_factor=2,
             mode='bilinear', align_corners=False)
        self.upsample4 = Interpolate(scale_factor=4,
             mode='bilinear', align_corners=False)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.pool4 = nn.MaxPool2d(4, stride=4)

        self.conv1 = UNetModule(in_channels, 32, bn=bn)
        self.conv2 = UNetModule(32, 64, bn=bn)
        self.conv3 = UNetModule(64, 128, bn=bn)
        self.conv4 = UNetModule(128, 256, bn=bn)
        self.center = UNetModule(256, 512, bn=bn)
        self.up4 = UNetModule(512 + 256, 256)
        self.up3 = UNetModule(256 + 128, 128)
        self.up2 = UNetModule(128 + 64, 64)
        self.up1 = UNetModule(64 + 32, 32)
        # final layer are logits
        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x, **kwargs):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        center = self.center(self.pool4(conv4))

        up4 = self.up4(torch.cat([conv4, self.upsample4(center)], 1))
        up3 = self.up3(torch.cat([conv3, self.upsample(up4)], 1))
        up2 = self.up2(torch.cat([conv2, self.upsample(up3)], 1))
        up1 = self.up1(torch.cat([conv1, self.upsample(up2)], 1))

        output = self.final(up1)
        return output


class DecoderModule(nn.Module):
    """
    DecoderModule for UNet11, UNet16

    Upsample version:
    [Interpolate(scale_factor, 'bilinear'),
    Con2dReLU(in_channels, mid_channels),
    Conv2dReLU(mid_channels, out_channels),
    ]

    DeConv version:
    [Con2dReLU(in_channels, mid_channels),
    ConvTranspose2d(mid_channels, out_channels, kernel=4, stride=2, pad=1),
    ReLU
    ]
    """
    def __init__(self, in_channels, mid_channels, out_channels, upsample=True):
        super(DecoderModule, self).__init__()
        if upsample:
            modules = OrderedDict([
                ('interpolate', Interpolate(scale_factor=2, mode='bilinear',
                    align_corners=False)),
                ('conv1', Conv2dReLU(in_channels, mid_channels)),
                ('conv2', Conv2dReLU(mid_channels, out_channels))
                ])
        else:
            modules = OrderedDict([
                ('conv', Conv2dReLU(in_channels, mid_channels)),
                ('deconv', nn.ConvTranspose2d(mid_channels,
                    out_channels, kernel_size=4, stride=2, padding=1)),
                ('relu', nn.ReLU(inplace=True))
                ])
        self.l = nn.Sequential(modules)

    def forward(self, x):
        return self.l(x)


class UNet11(nn.Module):
    """
    UNet11: use VGG11 as encoder and corresponding DecoderModule as decoder
    pretrained-model: ImageNet
    """
    def __init__(self, in_channels, num_classes, pretrained=True, bn=False, upsample=False):
        super(UNet11, self).__init__()
        if bn:
            self.vgg11 = models.vgg11_bn(pretrained=pretrained).features
            pool_idxs = [3, 7, 14, 21, 28]
        else:
            self.vgg11 = models.vgg11(pretrained=pretrained).features
            pool_idxs = [2, 5, 10, 15, 20]

        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = self.vgg11[0:pool_idxs[0]]
        self.conv2 = self.vgg11[pool_idxs[0]+1:pool_idxs[1]]
        self.conv3 = self.vgg11[pool_idxs[1]+1:pool_idxs[2]]
        self.conv4 = self.vgg11[pool_idxs[2]+1:pool_idxs[3]]
        self.conv5 = self.vgg11[pool_idxs[3]+1:pool_idxs[4]]

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(512 + 256, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(512 + 256, 512, 128, upsample=upsample)
        self.dec3 = DecoderModule(256 + 128, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(128 + 64, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(64 + 32, 32)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)


    def forward(self, x, **kwargs):
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.pool(conv1)) # 128
        conv3 = self.conv3(self.pool(conv2)) # 256
        conv4 = self.conv4(self.pool(conv3)) # 512
        conv5 = self.conv5(self.pool(conv4)) # 512
        center = self.center(self.pool(conv5)) # 256

        dec5 = self.dec5(torch.cat([center, conv5], 1)) # 256
        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) # 128
        dec3 = self.dec3(torch.cat([dec4, conv3], 1)) # 64
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) # 32
        dec1 = self.dec1(torch.cat([dec2, conv1], 1)) # 32

        output = self.final(dec1)
        return output



class UNet11_LSTM(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=True, bn=False, upsample=False):
        super(UNet11_LSTM, self).__init__()
        if bn:
            self.vgg11 = models.vgg11_bn(pretrained=pretrained).features
            pool_idxs = [3, 7, 14, 21, 28]
        else:
            self.vgg11 = models.vgg11(pretrained=pretrained).features
            pool_idxs = [2, 5, 10, 15, 20]

        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = self.vgg11[0:pool_idxs[0]]
        self.conv2 = self.vgg11[pool_idxs[0]+1:pool_idxs[1]]
        self.conv3 = self.vgg11[pool_idxs[1]+1:pool_idxs[2]]
        self.conv4 = self.vgg11[pool_idxs[2]+1:pool_idxs[3]]
        self.conv5 = self.vgg11[pool_idxs[3]+1:pool_idxs[4]]

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(512 + 256, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(512 + 256, 512, 128, upsample=upsample)
        self.dec3 = DecoderModule(256 + 128, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(128 + 64, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(64 + 32, 32)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)
        self.convlstm = ConvLSTM(input_size=(16,20),
                                 input_dim=512,
                                 hidden_dim=[512,512],
                                 kernel_size=(3,3),
                                 num_layers=2,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)

    def forward(self, x):
        x = torch.unbind(x, dim=1)
        data = []
        for item in x:
            conv1 = self.conv1(item)  # 64
            conv2 = self.conv2(self.pool(conv1))  # 128
            conv3 = self.conv3(self.pool(conv2))  # 256
            conv4 = self.conv4(self.pool(conv3))  # 512
            conv5 = self.conv5(self.pool(conv4))  # 512
            features = self.pool(conv5)
            data.append(features.unsqueeze(0))
        data = torch.cat(data, dim=0)
        lstm, _ = self.convlstm(data)
        test = lstm[0][ -1,:, :, :, :]
        center = self.center(test)   #maybe a bug
        dec5 = self.dec5(torch.cat([center, conv5], 1))  # 256
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))  # 128
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))  # 64
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))  # 32
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))  # 32
        output = self.final(dec1)
        return output


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda(),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda())


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor=input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])


                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output = layer_output.permute(1, 0, 2, 3, 4)

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
