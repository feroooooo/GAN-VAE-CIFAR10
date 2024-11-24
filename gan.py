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