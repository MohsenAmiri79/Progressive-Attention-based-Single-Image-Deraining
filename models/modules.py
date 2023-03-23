from utilities.evaluation import SSIM
from torch.autograd import Variable

import lightning.pytorch as pl
import torch.nn.functional as F
import torch.nn as nn
import torch


class PReNet(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input

        return x


class PReNet_r(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet_r, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h

            for j in range(5):
                resx = x
                x = F.relu(self.res_conv1(x) + resx)

            x = self.conv(x)
            x = input + x

        return x


class PReDecoderNet(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReDecoderNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.in_conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

        self.res_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 96, 4, 2, 1),
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.GELU(),
        )

        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Tanh())
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        for _ in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.in_conv(x)

            x = torch.cat((x, h), 1)
            c = self.conv_f(x) * c + self.conv_i(x) * self.conv_g(x)
            h = self.conv_o(x) * torch.tanh(c)

            x = h

            for _ in range(5):
                x = F.gelu(self.res_conv(x) + x)

            x = self.out_conv(x)
            x = input + x

        return x


class PReLSTMNet(nn.Module):
    def __init__(self, recurrent_iter=7, use_GPU=True):
        super(PReLSTMNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.in_conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

        self.res_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.GELU()
        )

        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Tanh())
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        for _ in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.in_conv(x)

            for _ in range(5):
                x = torch.cat((x, h), 1)
                c = self.conv_f(x) * c + self.conv_i(x) * self.conv_g(x)
                h = self.conv_o(x) * torch.tanh(c)
                x = F.gelu(self.res_conv(h) + h)

            x = self.out_conv(x)
            x = input + x

        return x


class PReComboNet(nn.Module):
    def __init__(self, recurrent_iter=7, use_GPU=True):
        super(PReComboNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.in_conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

        self.res_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 96, 4, 2, 1),
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.GELU(),
        )

        self.conv_i_outer = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_f_outer = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_g_outer = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Tanh())
        self.conv_o_outer = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())

        self.conv_i_inner = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_f_inner = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_g_inner = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Tanh())
        self.conv_o_inner = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        x = input
        h_outer = Variable(torch.zeros(batch_size, 32, row, col))
        c_outer = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h_outer = h_outer.cuda()
            c_outer = c_outer.cuda()

        for _ in range(self.iteration):
            h_inner = Variable(torch.zeros(batch_size, 32, row, col))
            c_inner = Variable(torch.zeros(batch_size, 32, row, col))

            if self.use_GPU:
                h_inner = h_inner.cuda()
                c_inner = c_inner.cuda()

            x = torch.cat((input, x), 1)
            x = self.in_conv(x)

            x = torch.cat((x, h_outer), 1)
            c_outer = self.conv_f_outer(
                x) * c_outer + self.conv_i_outer(x) * self.conv_g_outer(x)
            h_outer = self.conv_o_outer(x) * torch.tanh(c_outer)

            for i in range(5):
                x = torch.cat((h_outer, h_inner), 1)
                c_inner = self.conv_f_inner(
                    x) * c_inner + self.conv_i_inner(x) * self.conv_g_inner(x)
                h_inner = self.conv_o_inner(x) * torch.tanh(c_inner)
                x = F.gelu(self.res_conv(h_inner) + h_inner)

            x = self.out_conv(x)
            x = input + x

        return x


class PReHAYULANet(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReHAYULANet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.in_conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 96, 4, 2, 1),
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.GELU(),
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 96, 4, 2, 1),
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.GELU(),
        )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 96, 4, 2, 1),
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.GELU(),
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 96, 4, 2, 1),
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.GELU(),
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 96, 4, 2, 1),
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.GELU(),
        )

        self.res_convs = [self.res_conv1, self.res_conv2,
                          self.res_conv3, self.res_conv4, self.res_conv5]

        self.conv_i_outer = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_f_outer = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_g_outer = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Tanh())
        self.conv_o_outer = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())

        self.conv_i_inner = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_f_inner = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_g_inner = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Tanh())
        self.conv_o_inner = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        x = input
        h_outer = Variable(torch.zeros(batch_size, 32, row, col))
        c_outer = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h_outer = h_outer.cuda()
            c_outer = c_outer.cuda()

        for _ in range(self.iteration):
            h_inner = Variable(torch.zeros(batch_size, 32, row, col))
            c_inner = Variable(torch.zeros(batch_size, 32, row, col))

            if self.use_GPU:
                h_inner = h_inner.cuda()
                c_inner = c_inner.cuda()

            x = torch.cat((input, x), 1)
            x = self.in_conv(x)

            x = torch.cat((x, h_outer), 1)
            c_outer = self.conv_f_outer(
                x) * c_outer + self.conv_i_outer(x) * self.conv_g_outer(x)
            h_outer = self.conv_o_outer(x) * torch.tanh(c_outer)

            for i in range(5):
                x = torch.cat((h_outer, h_inner), 1)
                c_inner = self.conv_f_inner(
                    x) * c_inner + self.conv_i_inner(x) * self.conv_g_inner(x)
                h_inner = self.conv_o_inner(x) * torch.tanh(c_inner)
                x = F.gelu(self.res_convs[i](h_inner) + h_inner)

            x = self.out_conv(x)
            x = input + x

        return x


class PReAENet(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True, im_size=128):
        super(PReAENet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.in_conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding="same"),
            nn.GELU(),
            nn.MaxPool2d((2, 2), padding=0),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding="same"),
            nn.GELU(),
            nn.MaxPool2d((2, 2), padding=0),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding="same"),
            nn.GELU(),
            nn.MaxPool2d((2, 2), padding=0),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding="same"),
            nn.GELU(),
            nn.MaxPool2d((2, 2), padding=0)
        )

        self.transformer_layers = nn.TransformerEncoderLayer(
            d_model=128, nhead=4)
        self.transformer = nn.TransformerEncoder(
            self.transformer_layers, num_layers=4)
        self.transformer_input_size = im_size // 16

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(
                3, 3), stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(
                3, 3), stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 64, kernel_size=(
                3, 3), stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(
                3, 3), stride=2, padding=1, output_padding=1),
            nn.GELU(),
        )

        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Tanh())
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        for _ in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.in_conv(x)

            x = torch.cat((x, h), 1)
            c = self.conv_f(x) * c + self.conv_i(x) * self.conv_g(x)
            h = self.conv_o(x) * torch.tanh(c)

            x = h
            for _ in range(3):
                x = self.encoder(x)
                t = x.view(-1, 128, self.transformer_input_size **
                           2).permute(2, 0, 1)
                t = self.transformer(t)
                t = t.permute(
                    1, 2, 0).view(-1, 128, self.transformer_input_size, self.transformer_input_size)
                x = torch.cat((x, t), 1)
                x = self.decoder(x)

            x = torch.cat((x, h), 1)
            x = self.out_conv(x)
            x = input + x

        return x


class PReAEATNet(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True, im_size=128):
        super(PReAEATNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.in_conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.GELU()
        )

        self.out_conv = nn.Conv2d(64, 3, 3, 1, 1)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.GELU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.GELU(),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GELU(),
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.GELU(),
        )

        self.transformer_layers = nn.TransformerEncoderLayer(
            d_model=128, nhead=8)
        self.transformer = nn.TransformerEncoder(
            self.transformer_layers, num_layers=6)
        self.transformer_input_size = im_size // 16

        self.decoder1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ConvTranspose2d(256, 128, kernel_size=(
                3, 3), stride=2, padding=1, output_padding=1),
            nn.GELU(),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ConvTranspose2d(128, 64, kernel_size=(
                3, 3), stride=2, padding=1, output_padding=1),
            nn.GELU(),
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ConvTranspose2d(64, 32, kernel_size=(
                3, 3), stride=2, padding=1, output_padding=1),
            nn.GELU(),
        )

        self.decoder4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ConvTranspose2d(32, 32, kernel_size=(
                3, 3), stride=2, padding=1, output_padding=1),
            nn.GELU(),
        )

        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Tanh())
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        for _ in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.in_conv(x)

            x = torch.cat((x, h), 1)
            c = self.conv_f(x) * c + self.conv_i(x) * self.conv_g(x)
            h = self.conv_o(x) * torch.tanh(c)

            x = h

            for i in range(3):
                x1 = self.encoder1(x)
                x2 = self.encoder2(x1)
                x3 = self.encoder3(x2)
                x4 = self.encoder4(x3)
                t = x4.view(-1, 128, self.transformer_input_size ** 2).permute(2, 0, 1)
                t = self.transformer(t)
                t = t.permute(1, 2, 0).view(-1, 128, self.transformer_input_size, self.transformer_input_size)
                x = torch.cat((t, x4), 1)
                x = self.decoder1(x)
                x = x + x3
                x = self.decoder2(x)
                x = x + x2
                x = self.decoder3(x)
                x = x + x1
                x = self.decoder4(x)

            x = torch.cat((x, h), 1)
            x = self.out_conv(x)
            x = x + input

        return x


class PReAENetPL(pl.LightningModule):
    def __init__(self, device, epochs=100, lr=1e-3, recurrent_iter=6, use_GPU=True, im_size=128):
        super(PReAENetPL, self).__init__()
        self.model = PReAENet(recurrent_iter, use_GPU, im_size).to(device)
        self.device_ = device
        self.criterion = SSIM()
        self.lr = lr
        self.epochs = epochs

    def training_step(self, batch, batch_idx):
        input, y = batch
        output = self.model(input.to(self.device_))
        loss = - self.criterion(y.to(self.device_), output)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, y = batch
        output = self.model(input.to(self.device_))
        loss = - self.criterion(y.to(self.device_), output)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)


class PReDeconvNetPL(pl.LightningModule):
    def __init__(self, device, epochs=100, lr=1e-3, recurrent_iter=6, use_GPU=True, im_size=128):
        super(PReDeconvNetPL, self).__init__()
        self.model = PReDecoderNet(recurrent_iter, use_GPU).to(device)
        self.device_ = device
        self.criterion = SSIM()
        self.lr = lr
        self.epochs = epochs

    def training_step(self, batch, batch_idx):
        input, y = batch
        output = self.model(input.to(self.device_))
        loss = - self.criterion(y.to(self.device_), output)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, y = batch
        output = self.model(input.to(self.device_))
        loss = - self.criterion(y.to(self.device_), output)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)


class EnDecoNet(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(EnDecoNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.in_conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

        self.encoder_decoder = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, groups=8),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1, groups=8),
            nn.Conv2d(64, 96, 3, 2, 1, groups=8),
            nn.GELU(),
            nn.ConvTranspose2d(96, 64, 3, 2, 1, output_padding=1, groups=8),
            nn.Conv2d(64, 64, 3, 1, 1, groups=8),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1, groups=8),
            nn.GELU(),
        )

        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Tanh())
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        for _ in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.in_conv(x)

            x = torch.cat((x, h), 1)
            c = self.conv_f(x) * c + self.conv_i(x) * self.conv_g(x)
            h = self.conv_o(x) * torch.tanh(c)

            x = h

            for _ in range(5):
                x = F.gelu(self.encoder_decoder(x) + x)

            x = self.out_conv(x)
            x = input + x

        return x

