from torch import nn
import torch.nn.functional as F


# 构造生成器
class GeneratorSimple(nn.Module):
    def __init__(self):
        super(GeneratorSimple, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )

        self.fc5 = nn.Sequential(
            nn.Linear(2048, 3072),  # 3072 = 3 x 32 x 32
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = x.reshape([-1, 3, 32, 32])  # CIFAR10是3通道,32x32大小
        return x


# 构造判别器
class DiscriminatorSimple(nn.Module):
    def __init__(self):
        super(DiscriminatorSimple, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(3072, 1024),  # 3072 = 3 x 32 x 32
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape([-1, 3072])  # 展平为3072维向量
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        input = input.unsqueeze(2).unsqueeze(3)
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)
