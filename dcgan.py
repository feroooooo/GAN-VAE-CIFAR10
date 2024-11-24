from torch import nn
import torch.nn.functional as F
    
# 构造生成器  
class GeneratorCIFAR10(nn.Module):
    def __init__(self):
        super(GeneratorCIFAR10, self).__init__()
        self.conv_1 = nn.ConvTranspose2d(
            100,512,4,1,0,
            bias=False
            )
        nn.init.normal_(self.conv_1.weight, mean=0.0, std=0.02)
        
        self.bn_1 = nn.BatchNorm2d(
            512,
            momentum=0.8
            )
        nn.init.normal_(self.bn_1.weight, mean=1.0, std=0.02)
        
        self.conv_2 = nn.ConvTranspose2d(
            512,256,4,2,1,
            bias=False
            )
        nn.init.normal_(self.conv_2.weight, mean=0.0, std=0.02)
        
        self.bn_2 = nn.BatchNorm2d(
            256,
            momentum=0.8
            )
        nn.init.normal_(self.bn_2.weight, mean=1.0, std=0.02)
        
        self.conv_3 = nn.ConvTranspose2d(
            256,128,4,2,1,
            bias=False
            )
        nn.init.normal_(self.conv_3.weight, mean=0.0, std=0.02)
        
        self.bn_3 = nn.BatchNorm2d(
            128,
            momentum=0.8
            )
        nn.init.normal_(self.bn_3.weight, mean=1.0, std=0.02)
        
        self.conv_4 = nn.ConvTranspose2d(
            128,64,4,2,1,
            bias=False
            )
        nn.init.normal_(self.conv_4.weight, mean=0.0, std=0.02)
        
        self.bn_4 = nn.BatchNorm2d(
            64,
            momentum=0.8
            )
        nn.init.normal_(self.bn_4.weight, mean=1.0, std=0.02)
        
        self.conv_5 = nn.Conv2d(
            64,3,5,1,0,
            bias=False
            )
        nn.init.normal_(self.conv_2.weight, mean=0.0, std=0.02)
        
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = x.reshape([-1, 100, 1, 1])
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = F.relu(x)
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = F.relu(x)
        x = self.conv_5(x)
        x = self.tanh(x)
        return x


# 构造判别器
class DiscriminatorCIFAR10(nn.Module):
    def __init__(self):
        super(DiscriminatorCIFAR10, self).__init__()
        self.conv_1 = nn.Conv2d(
            3,64,3,1,1,
            bias=False
            )
        nn.init.normal_(self.conv_1.weight, mean=0.0, std=0.02)
        
        self.conv_2 = nn.Conv2d(
            64,128,4,2,1,
            bias=False
            )
        nn.init.normal_(self.conv_2.weight, mean=0.0, std=0.02)
        
        self.bn_2 = nn.BatchNorm2d(
            128,
            momentum=0.8
            )
        nn.init.normal_(self.bn_2.weight, mean=1.0, std=0.02)
        
        self.conv_3 = nn.Conv2d(
            128,256,4,2,1,
            bias=False
            )
        nn.init.normal_(self.conv_3.weight, mean=0.0, std=0.02)
        
        self.bn_3 = nn.BatchNorm2d(
            256,
            momentum=0.8
            )
        nn.init.normal_(self.bn_3.weight, mean=1.0, std=0.02)
        
        self.conv_4 = nn.Conv2d(
            256,512,4,1,0,
            bias=False
            )
        nn.init.normal_(self.conv_4.weight, mean=0.0, std=0.02)
        
        self.bn_4 = nn.BatchNorm2d(
            512,
            momentum=0.8
            )
        nn.init.normal_(self.bn_4.weight, mean=1.0, std=0.02)
        
        self.conv_5 = nn.Conv2d(
            512,1,4,1,0,
            bias=False
            )
        nn.init.normal_(self.conv_5.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        x = self.conv_1(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        
        x = self.conv_5(x)
        x = F.sigmoid(x[0][0][0])
        return x