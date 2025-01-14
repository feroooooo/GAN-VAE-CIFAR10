import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到 [-1, 1]
])

# 下载并加载训练数据
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

# 下载并加载测试数据
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 8, 8]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 4, 4]
            nn.ReLU(),
            nn.Flatten()  # [B, 128*4*4 = 2048]
        )
        # 潜在空间的均值和对数方差
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)
        
        # 解码器
        self.decoder_input = nn.Linear(latent_dim, 2048)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),  # [B, 128, 4, 4]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # [B, 3, 32, 32]
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)    # 采样ε
        return mu + eps * std          # 重参数化
    
    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(latent_dim=128).to(device)

# 定义损失函数
def loss_function(recon_x, x, mu, logvar):
    # 重建损失
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

num_epochs = 30
model.train()
for epoch in range(num_epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # 每个epoch后展示重建图像
    with torch.no_grad():
        model.eval()
        test_img, _ = next(iter(test_loader))
        test_img = test_img.to(device)
        recon, _, _ = model(test_img)
        model.train()
        
        # 将图像从 [-1, 1] 转换回 [0, 1]
        test_img = test_img.cpu().numpy()
        recon = recon.cpu().numpy()
        
        # 显示原图和重建图
        n = 10  # 展示的图像数量
        fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(20, 4))
        for i in range(n):
            # 原图
            axes[0, i].imshow(np.transpose((test_img[i] * 0.5 + 0.5), (1, 2, 0)))
            axes[0, i].axis('off')
            # 重建图
            axes[1, i].imshow(np.transpose((recon[i] * 0.5 + 0.5), (1, 2, 0)))
            axes[1, i].axis('off')
        plt.suptitle(f'Epoch {epoch+1}')
        plt.show()

# 生成新图像
model.eval()
with torch.no_grad():
    # 从标准正态分布中采样潜在向量
    z = torch.randn(10, 128).to(device)
    generated = model.decode(z)
    generated = generated.cpu().numpy()
    
    # 显示生成的图像
    fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(20, 2))
    for i in range(10):
        axes[i].imshow(np.transpose((generated[i] * 0.5 + 0.5), (1, 2, 0)))
        axes[i].axis('off')
    plt.suptitle('Generated Images')
    plt.show()
