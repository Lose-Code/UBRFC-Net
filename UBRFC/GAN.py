import torch
import functools
from torch import nn
#from torchsummary import summary

from Attention import Attention
from Util import PALayer, ConvGroups, FE_Block, Fusion_Block, ResnetBlock, ConvBlock, CALayer


class Generator(nn.Module):
    def __init__(self, ngf=64, bn=False):
        super(Generator, self).__init__()
        # 下采样
        self.down1 = ResnetBlock(3, first=True)

        self.down2 = ResnetBlock(ngf, levels=2)

        self.down3 = ResnetBlock(ngf * 2, levels=2, bn=bn)

        self.res = nn.Sequential(
            ResnetBlock(ngf * 4, levels=6, down=False, bn=True)
        )

        # 上采样

        self.up1 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.InstanceNorm2d(ngf * 2) if not bn else nn.BatchNorm2d(ngf * 2),
        )

        self.up2 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
        )

        self.info_up1 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.InstanceNorm2d(ngf * 2) if not bn else nn.BatchNorm2d(ngf * 2, eps=1e-5),
        )

        self.info_up2 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf)  # if not bn else nn.BatchNorm2d(ngf, eps=1e-5),
        )

        self.up3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0),
            nn.Tanh())

        self.pa2 = PALayer(ngf * 4)
        self.pa3 = PALayer(ngf * 2)
        self.pa4 = PALayer(ngf)

        # self.ca2 = CALayer(ngf * 4)
        # self.ca3 = CALayer(ngf * 2)
        # self.ca4 = CALayer(ngf)
        self.ca2 = Attention(ngf * 4)
        self.ca3 = Attention(ngf * 2)
        self.ca4 = Attention(ngf)

        self.down_dcp = ConvGroups(3, bn=bn)

        self.fam1 = FE_Block(ngf, ngf)
        self.fam2 = FE_Block(ngf, ngf * 2)
        self.fam3 = FE_Block(ngf * 2, ngf * 4)

        self.att1 = Fusion_Block(ngf)
        self.att2 = Fusion_Block(ngf * 2)
        self.att3 = Fusion_Block(ngf * 4, bn=bn)

        self.merge2 = nn.Sequential(
            ConvBlock(ngf * 2, ngf * 2, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        )
        self.merge3 = nn.Sequential(
            ConvBlock(ngf, ngf, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, hazy, img=0, first=True):
        if not first:
            dcp_down1, dcp_down2, dcp_down3 = self.down_dcp(img)
        x_down1 = self.down1(hazy)  # [bs, ngf, ngf * 4, ngf * 4][1,64,256,256]

        att1 = self.att1(dcp_down1, x_down1) if not first else x_down1 #[1,64,256,256]

        x_down2 = self.down2(x_down1)  # [bs, ngf*2, ngf*2, ngf*2][1,128,128,128]
        att2 = self.att2(dcp_down2, x_down2) if not first else None #None
        fuse2 = self.fam2(att1, att2) if not first else self.fam2(att1, x_down2)#[1,128,128,128]

        x_down3 = self.down3(x_down2)  # [bs, ngf * 4, ngf, ngf]#[1,256,64,64]
        att3 = self.att3(dcp_down3, x_down3) if not first else None #None
        fuse3 = self.fam3(fuse2, att3) if not first else self.fam3(fuse2, x_down3)#[1,256,64,64]

        x6 = self.pa2(self.ca2(self.res(x_down3)))#[1,256,64,64]

        fuse_up2 = self.info_up1(fuse3)#[1,128,128,128]
        fuse_up2 = self.merge2(fuse_up2 + x_down2)#[1,128,128,128]

        fuse_up3 = self.info_up2(fuse_up2)#[1,64,256,256]
        fuse_up3 = self.merge3(fuse_up3 + x_down1)#[1,64,256,256]

        x_up2 = self.up1(x6 + fuse3)#[1,128,128,128]
        x_up2 = self.ca3(x_up2)#[1,128,128,128]
        x_up2 = self.pa3(x_up2)#[1,128,128,128]

        x_up3 = self.up2(x_up2 + fuse_up2)#[1,64,256,256]
        x_up3 = self.ca4(x_up3)#[1,64,256,256]
        x_up3 = self.pa4(x_up3)#[1,64,256,256]

        x_up4 = self.up3(x_up3 + fuse_up3)#[1,3,256,256]

        return x_up4


class Discriminator(nn.Module):
    """
    Discriminator class
    """

    def __init__(self, inp=3, out=1):
        """
        Initializes the PatchGAN model with 3 layers as discriminator

        Args:
        inp: number of input image channels
        out: number of output image channels
        """

        super(Discriminator, self).__init__()

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

        model = [
            nn.Conv2d(inp, 64, kernel_size=4, stride=2, padding=1),  # input 3 channels
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            norm_layer(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            norm_layer(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=True),
            norm_layer(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, out, kernel_size=4, stride=1, padding=1)  # output only 1 channel (prediction map)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """
            Feed forward the image produced by generator through discriminator

            Args:
            input: input image

            Returns:
            outputs prediction map with 1 channel
        """
        result = self.model(input)

        return result


class discriminator(nn.Module):
    def __init__(self, bn=False, ngf=64):
        super(discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, padding=0),
            nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, padding=0),
            nn.InstanceNorm2d(ngf * 4) if not bn else nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf * 4) if not bn else nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, padding=0),
            nn.BatchNorm2d(ngf * 8) if bn else nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(ngf * 8) if bn else nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ngf * 8, ngf * 16, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ngf * 16, 1, kernel_size=1)

        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))




if __name__ == '__main__':
    net = Generator()
    #summary(net, (3,256,256), batch_size=1, device="cpu")
    x = torch.randn((1,3,256,256))
    #
    #print(net)
    y = net(x)
    print(y.size())