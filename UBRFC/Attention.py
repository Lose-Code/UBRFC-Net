import math
import torch
from torch import nn
#from torchstat import stat  # 查看网络参数

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out




# class Attention(nn.Module):
    # def __init__(self,channel,b=1, gamma=2):
        # super(Attention, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)#全局平均池化
        # #一维卷积
        # t = int(abs((math.log(channel, 2) + b) / gamma))
        # k = t if t % 2 else t + 1
        # self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        # self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        # self.sigmoid = nn.Sigmoid()
        # self.mix = Mix()
        # #全连接
        # #self.fc = nn.Linear(channel,channel)
        # #self.softmax = nn.Softmax(dim=1)

    # def forward(self, input):
        # x = self.avg_pool(input)
        # x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # out1 = self.sigmoid(x1)
        # out2 = self.fc(x)
        # out2 = self.sigmoid(out2)
        # out = self.mix(out1,out2)
        # out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # #out = self.softmax(out)

        # return input*out
class Attention(nn.Module):
    def __init__(self,channel,b=1, gamma=2):
        super(Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#全局平均池化
        #一维卷积
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()


    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)#(1,64,1)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)#(1,1,64)
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)#(1,64,1,1)
        #x1 = x1.transpose(-1, -2).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)

        #out2 = self.fc(x)
        out2 = self.sigmoid(out2)
        out = self.mix(out1,out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)

        return input*out

if __name__ == '__main__':
    input = torch.rand(1,64,256,256)

    A = Attention(channel=64)
    #stat(A, input_size=[64, 1, 1])
    y = A(input)
    print(y.size())


