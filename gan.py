from torch import nn

# 构造生成器  
class GeneratorCIFAR10(nn.Module):
    def __init__(self):
        super(GeneratorCIFAR10, self).__init__()
        
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
class DiscriminatorCIFAR10(nn.Module):
    def __init__(self):
        super(DiscriminatorCIFAR10, self).__init__()
        
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
    

class GeneratorDCGAN(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
       """
       参数说明：
       nz: 噪声向量的维度
       ngf: 生成器特征图的基础通道数
       nc: 输出图像的通道数(CIFAR10为3)
       """
       super(GeneratorDCGAN, self).__init__()
       
       self.main = nn.Sequential(
           # 输入是 nz x 1 x 1
           nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
           nn.BatchNorm2d(ngf * 8),
           nn.ReLU(True),
           # 状态尺寸: (ngf*8) x 4 x 4
           
           nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
           nn.BatchNorm2d(ngf * 4),
           nn.ReLU(True),
           # 状态尺寸: (ngf*4) x 8 x 8
           
           nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
           nn.BatchNorm2d(ngf * 2),
           nn.ReLU(True),
           # 状态尺寸: (ngf*2) x 16 x 16
           
           nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
           nn.Tanh()
           # 最终尺寸: nc x 32 x 32
       )
    
    def forward(self, x):
       # 输入向量需要重塑为 (batch_size, nz, 1, 1)
       x = x.view(x.size(0), -1, 1, 1)
       return self.main(x)
   

class DiscriminatorDCGAN(nn.Module):
    def __init__(self, nc=3, ndf=64):
       """
       参数说明：
       nc: 输入图像的通道数
       ndf: 判别器特征图的基础通道数
       """
       super(DiscriminatorDCGAN, self).__init__()
       
       self.main = nn.Sequential(
           # 输入是 nc x 32 x 32
           nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
           nn.LeakyReLU(0.2, inplace=True),
           # 状态尺寸: ndf x 16 x 16
           
           nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
           nn.BatchNorm2d(ndf * 2),
           nn.LeakyReLU(0.2, inplace=True),
           # 状态尺寸: (ndf*2) x 8 x 8
           
           nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
           nn.BatchNorm2d(ndf * 4),
           nn.LeakyReLU(0.2, inplace=True),
           # 状态尺寸: (ndf*4) x 4 x 4
           
           nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
           nn.Sigmoid()
           # 输出尺寸: 1 x 1 x 1
       )
    def forward(self, x):
       return self.main(x).view(-1, 1).squeeze(1)