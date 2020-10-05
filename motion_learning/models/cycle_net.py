from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import init
import os
import torchvision
import sys
from models.model_utils import conv2d, deconv2d, gram, gram_p, backWarp, getFlowCoeff, up, down, UNet
os.environ['TORCH_HOME'] = '../pretrained_model'
from models.vgg import Vgg16
sys.path.append("../")
from flownet2_pytorch.models import FlowNet2
from flownet2_pytorch.networks.resample2d_package.resample2d import Resample2d
import cv2
import numpy as np
import torch.nn.functional as F


class CycleNet(nn.Module):
    def __init__(self, args, is_training = True):
        super(CycleNet,self).__init__()
        self.sequence_length = args.sequence_length
        self.flowComp = UNet(6, 4)
        self.ArbTimeFlowIntrp = UNet(30, 8)
        if is_training:
            self.back_warp = backWarp(448, 448)
        else:
            self.back_warp = backWarp(1280, 1024)
        self.L1Loss = nn.L1Loss()
        self.MSEloss = nn.MSELoss()
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16_conv_4_3 = nn.Sequential(*list(self.vgg16.children())[0][:22])
        self.vgg16_conv_4_3.cuda()
        for param in self.vgg16_conv_4_3.parameters():
            param.requires_grad = False
        return

    def interframe_optical_flow(self, input_images):
        input_images = [torch.flip(input_image, dims=[1]) for input_image in input_images]

        # Create image pairs for flownet, then merge batch and frame dimension
        # so theres only a single call to flownet2 is done.
        flownet2_inputs = torch.stack(
            [torch.cat([input_images[i + 1].unsqueeze(2), input_images[i].unsqueeze(2)], dim=2) for i in
             range(0, self.sequence_length - 1)], dim=0).contiguous()

        batch_size, channel_count, height, width = input_images[0].shape
        flownet2_inputs_flattened = flownet2_inputs.view(-1, channel_count, 2, height, width)
        flownet2_outputs = [self.flownet2(flownet2_input) for flownet2_input in torch.chunk(flownet2_inputs_flattened, self.sequence_length - 1)]
        # FIXME: flipback images to BGR,
        # input_images = [torch.flip(input_image, dims=[1]) for input_image in input_images]

        return flownet2_outputs

    def flow_to_arrow(self, flow_uv, positive=False):
        '''
        Expects a two dimensional flow image of shape [H,W,2]

        :param flow_uv: np.ndarray of shape [H,W,2]
        :param clip_flow: float, maximum clipping value for flow
        :return: flow image shown in arrow
        '''
        flow_uv = flow_uv.data.cpu().numpy().squeeze().transpose(1, 2, 0)
        h, w = flow_uv.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        new_x = np.rint(x + flow_uv[:, :, 0]).astype(dtype=np.int64)
        new_y = np.rint(y + flow_uv[:, :, 1]).astype(dtype=np.int64)
        # clip to the boundary
        new_x = np.clip(new_x, 0, w)
        new_y = np.clip(new_y, 0, h)
        # empty image
        coords_origin = np.array([x.flatten(), y.flatten()]).T
        coords_new = np.array([new_x.flatten(), new_y.flatten()]).T

        flow_arrow = np.ones((h, w, 3), np.uint8) * 255
        for i in range(0, len(coords_origin), 1000):
            if positive:
                cv2.arrowedLine(flow_arrow, tuple(coords_origin[i]), tuple(coords_new[i]), (255, 0, 0), 2)
            else:
                cv2.arrowedLine(flow_arrow, tuple(coords_new[i]), tuple(coords_origin[i]), (255, 0, 0), 2)
        return flow_arrow


    def network_output(self, I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, F_1_t, F_0_t, g_I1_F_t_1, g_I0_F_t_0, g_It_F_1_t, g_It_F_0_t):

        # Network input

        x= torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, F_1_t, F_0_t, g_I1_F_t_1, g_I0_F_t_0, g_It_F_1_t, g_It_F_0_t), dim=1)

        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)

        return x

    def normalize(self, tensorInput):
        tensorInput = tensorInput / 255
        tensorRed = (tensorInput[:, 0:1, :, :] - 0.479) / 0.199
        tensorGreen = (tensorInput[:, 1:2, :, :] - 0.302) / 0.169
        tensorBlue = (tensorInput[:, 2:3, :, :] - 0.341) / 0.188
        return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)


    def prepare_inputs(self, input_dict):
        images = input_dict['image']  # expects a list

        # input_images = images[:-1]
        # I0 = self.normalize(images[0])
        # I1 = self.normalize(images[1])
        # input_images = [I0, I1]
        I0 = images[0] /255
        I1 = images[1] /255
        # input_images = [I0, I1, I0]

        return I0, I1
    def forward(self, input_dict, label_image=None):

        I0, I1 = self.prepare_inputs(input_dict)
        # Calculate flow between reference frames I0 and I1
        #flows = self.interframe_optical_flow(input_images)
        flowOut = self.flowComp(torch.cat((I0, I1), dim=1))

        F_0_1 = flowOut[:, :2, :, :]
        F_1_0 = flowOut[:, 2:, :, :]


        fCoeff = getFlowCoeff()
        # Calculate intermediate flows
        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0
        F_1_t = F_1_0 - F_t_0
        F_0_t = F_0_1 - F_t_1


        # Get intermediate frames from the intermediate flows
        g_I0_F_t_0 = self.back_warp(I0, F_t_0)
        g_I1_F_t_1 = self.back_warp(I1, F_t_1)
        g_It_F_1_t = self.back_warp(g_I0_F_t_0, F_1_t) # pred I1
        g_It_F_0_t = self.back_warp(g_I1_F_t_1, F_0_t) # pred I0

        intrpOut = self.ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, F_1_t, F_0_t, g_I1_F_t_1, g_I0_F_t_0, g_It_F_1_t, g_It_F_0_t), dim=1))

        # Extract optical flow residuals and visibility maps
        F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        F_1_t_f = intrpOut[:, 4:6, :, :] + F_1_t
        F_0_t_f = intrpOut[:, 6:8, :, :] + F_0_t
        # V_t_0 = F.sigmoid(intrpOut[:, 8:9, :, :])
        # V_t_1 = 1 - V_t_0

        It_0 = self.back_warp(I0, F_t_0_f)
        It_1 = self.back_warp(I1, F_t_1_f)
        I0_t = self.back_warp(It_1, F_0_t_f)
        I1_t = self.back_warp(It_0, F_1_t_f)



        # calculate losses
        losses = {}

        losses['reconstruct'] = self.L1Loss(It_0, It_1)

        losses['perceptual'] = self.MSEloss(self.vgg16_conv_4_3(It_0), self.vgg16_conv_4_3(It_1))

        losses['warp'] = self.L1Loss(I0, I0_t) + self.L1Loss(I1, I1_t)
        loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(
            torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
        loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(
            torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
        losses['smooth'] = loss_smooth_0_1 + loss_smooth_1_0
        # print(2.0*losses['reconstruct'], 0.02*losses['perceptual'],  1.0*losses['warp'], 1.5*losses['smooth'])
        losses['tot'] = 2.0*losses['reconstruct'] + 0.02*losses['perceptual']+ 1.0*losses['warp'] +1.0*losses['smooth']
        return losses, It_0*255, It_1*255
