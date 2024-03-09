#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import config
import os
import numpy as np
import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class UNET_LSTM_BIDIRECTIONAL_AUTOENCODER_TACA_4(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_clusters=None):
        super(UNET_LSTM_BIDIRECTIONAL_AUTOENCODER_TACA_4, self).__init__()

        self.conv1_1 = torch.nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.conv4_1 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv4_2 = torch.nn.Conv2d(128, 128, 3, padding=1)
        # self.conv5_1 = torch.nn.Conv2d(128, 256, 3, padding=1)
        # self.conv5_2 = torch.nn.Conv2d(256, 256, 3, padding=1)

        # self.unpool4 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.upconv4_1 = torch.nn.Conv2d(256, 128, 3, padding=1)
        # self.upconv4_2 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.unpool3 = torch.nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.upconv3_1 = torch.nn.Conv2d(128, 64, 3, padding=1)
        self.upconv3_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.unpool2 = torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv2_1 = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.upconv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.unpool1 = torch.nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.upconv1_1 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.upconv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)

        self.out = torch.nn.Conv2d(16, out_channels, kernel_size=1, padding=0)

        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.0)

        self.lstm = torch.nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.attention = torch.nn.Linear(256, 1)
        # self.attention_ca = torch.nn.Linear(config.time_steps * config.input_patch_size * config.input_patch_size, 1) #c
        # self.unpool4 = torch.nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2)

        self.lstm_decoder = torch.nn.LSTM(256, 256, batch_first=True)

        if n_clusters is not None:
            self.alpha = 1.0
            self.clusterCenter = torch.nn.Parameter(torch.zeros(n_clusters, 512))
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        batch_size, seq_len, channels, input_patch_size, input_patch_size = x.shape

        x = x.permute(0, 2, 1, 3, 4)
        x_reshape = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3] * x.shape[4])
        x_reshape = x_reshape.reshape(x_reshape.shape[0] * x_reshape.shape[1], x_reshape.shape[2])
        tanh = torch.tanh(x_reshape)
        attention_ca = torch.nn.Linear(seq_len * config.input_patch_size * config.input_patch_size, 1) #c
        attn = attention_ca(tanh).view(-1, channels, 1, 1)
        attn_ca = torch.nn.functional.softmax(torch.squeeze(torch.nn.functional.avg_pool2d(attn, 1)), dim=1)
        x = (attn_ca.view(-1, 1) * x_reshape).view(-1, channels,
                                                   seq_len * config.input_patch_size * config.input_patch_size).view(
            -1, channels, seq_len, config.input_patch_size, config.input_patch_size).permute(0, 2, 1, 3, 4)

        x = x.reshape(-1, channels, input_patch_size, input_patch_size)

        conv1 = self.relu(self.conv1_2(self.relu(self.conv1_1(x))))
        maxpool1 = self.maxpool(conv1)
        maxpool1 = self.dropout(maxpool1)

        conv2 = self.relu(self.conv2_2(self.relu(self.conv2_1(maxpool1))))
        maxpool2 = self.maxpool(conv2)
        maxpool2 = self.dropout(maxpool2)

        conv3 = self.relu(self.conv3_2(self.relu(self.conv3_1(maxpool2))))
        maxpool3 = self.maxpool(conv3)
        maxpool3 = self.dropout(maxpool3)

        conv4 = self.relu(self.conv4_2(self.relu(self.conv4_1(maxpool3))))
        # maxpool4 = self.maxpool(conv4)
        conv4 = self.dropout(conv4)

        # conv5 = self.relu(self.conv5_2(self.relu(self.conv5_1(maxpool4))))
        # conv5 = self.dropout(conv5)

        shape_enc = conv4.shape
        conv4 = conv4.view(-1, seq_len, conv4.shape[1], conv4.shape[2] * conv4.shape[3])
        conv4 = conv4.permute(0, 3, 1, 2)
        conv4 = conv4.reshape(conv4.shape[0] * conv4.shape[1], seq_len, 128)
        lstm, _ = self.lstm(conv4)
        lstm = self.relu(lstm.reshape(-1, 256))
        attention_weights = torch.nn.functional.softmax(torch.squeeze(torch.nn.functional.avg_pool2d(
            self.attention(torch.tanh(lstm)).view(-1, shape_enc[2], shape_enc[3], seq_len).permute(0, 3, 1,
                                                                                                             2),
            shape_enc[2])), dim=1)
        context = torch.sum((attention_weights.view(-1, 1, 1, seq_len).repeat(1, shape_enc[2], shape_enc[3],
                                                                                        1).view(-1, 1) * lstm).view(-1,
                                                                                                                    seq_len,
                                                                                                                    256),
                            dim=1).view(-1, shape_enc[2], shape_enc[3], 256).permute(0, 3, 1, 2)
        attention_weights_fixed = attention_weights.detach()
        context = self.dropout(context)

        code_vec = torch.nn.functional.avg_pool2d(context, context.shape[-1]).squeeze()
        # code_vec = torch.nn.functional.normalize(code_vec, p=2.0, dim=1, eps=1e-12)

        context = context.permute(0, 2, 3, 1)
        context = context.view(-1, context.shape[-1])
        out = torch.zeros(context.shape[0], seq_len, context.shape[-1]).to('cuda')
        input = torch.unsqueeze(context, dim=1)
        h = (torch.unsqueeze(torch.zeros_like(context), dim=0), torch.unsqueeze(torch.zeros_like(context), dim=0))
        for step in range(seq_len):
            output, h = self.lstm_decoder(input, h)
            out[:, step] = output.squeeze()
            input = torch.unsqueeze(output.squeeze() + context, dim=1)
        out = out.view(batch_size, -1, seq_len, out.shape[-1])
        out = out.permute(0, 2, 3, 1)
        out = out.view(batch_size, seq_len, out.shape[2], shape_enc[2], shape_enc[3])
        out = out.reshape(-1, out.shape[2], out.shape[3], out.shape[4])

        # unpool4 = self.unpool4(out)
        # upconv4 = self.relu(self.upconv4_2(unpool4))
        # upconv4 = self.dropout(upconv4)

        unpool3 = self.unpool3(out)
        upconv3 = self.relu(self.upconv3_2(unpool3))
        upconv3 = self.dropout(upconv3)

        unpool2 = self.unpool2(upconv3)
        upconv2 = self.relu(self.upconv2_2(unpool2))
        upconv2 = self.dropout(upconv2)

        unpool1 = self.unpool1(upconv2)
        upconv1 = self.relu(self.upconv1_2(unpool1))
        upconv1 = self.dropout(upconv1)

        out = self.out(upconv1)
        out = out.view(batch_size, seq_len, channels, input_patch_size, input_patch_size)
        return code_vec, out, attention_weights_fixed, attn_ca

    def updateClusterCenter(self, cc):
        self.clusterCenter.data = torch.from_numpy(cc).to('cuda')

    def getTDistribution(self, code_vec):
        xe = torch.unsqueeze(code_vec, 1).to('cuda') - self.clusterCenter.to('cuda')
        q = 1.0 / (1.0 + (torch.sum(torch.mul(xe, xe), 2) / self.alpha))
        q = q ** (self.alpha + 1.0) / 2.0
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def getTargetDistribution(self, q):
        weight = q ** 2 / q.sum(0)
        return torch.autograd.Variable((weight.t() / weight.sum(1)).t().data, requires_grad=True)