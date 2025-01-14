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
train_dataset = torchvision.datasets.CIFAR10(root='D:/data', train=True,
                                             download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 下载并加载测试数据
test_dataset = torchvision.datasets.CIFAR10(root='D:/data', train=False,
                                            download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # [B, 16, 16, 16]
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # [B, 32, 8, 8]
            nn.ReLU(True),
            nn.Conv2d(32, 64, 7)  # [B, 64, 2, 2] -> [B, 64, 1, 1]
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # [B, 32, 7, 7]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # [B, 16, 14, 14]
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # [B, 3, 28, 28]
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

num_epochs = 20
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data  # 我们只需要图像，不需要标签
        img = img.to(device)
        
        # 前向传播
        output = model(img)
        loss = criterion(output, img)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 打印损失
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 保存重建的图像
    with torch.no_grad():
        # 获取测试集中的一批图像
        test_img, _ = next(iter(test_loader))
        test_img = test_img.to(device)
        reconstructed = model(test_img)
        # 将图像从 [-1, 1] 转换回 [0, 1]
        test_img = test_img.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()
        
        # 显示原图和重建图
        fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(15, 4))
        for i in range(10):
            # 原图
            axes[0, i].imshow(np.transpose((test_img[i] * 0.5 + 0.5), (1, 2, 0)))
            axes[0, i].axis('off')
            # 重建图
            axes[1, i].imshow(np.transpose((reconstructed[i] * 0.5 + 0.5), (1, 2, 0)))
            axes[1, i].axis('off')
        plt.show()

# 示例：从测试集中选择两个图像，进行线性插值
with torch.no_grad():
    img1, _ = next(iter(test_loader))
    img2, _ = next(iter(test_loader))
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    encoded1 = model.encoder(img1)
    encoded2 = model.encoder(img2)
    
    # 线性插值
    alpha = 0.5
    interpolated = encoded1 * alpha + encoded2 * (1 - alpha)
    generated = model.decoder(interpolated)
    
    generated = generated.cpu().numpy()
    
    # 显示生成的图像
    fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(15, 2))
    for i in range(10):
        axes[i].imshow(np.transpose((generated[i] * 0.5 + 0.5), (1, 2, 0)))
        axes[i].axis('off')
    plt.show()
