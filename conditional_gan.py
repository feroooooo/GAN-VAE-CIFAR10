import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义超参数
num_epochs = 50
batch_size = 128
learning_rate = 0.0002
nz = 100  # 噪声向量维度
num_classes = 10  # CIFAR-10 有 10 个类别
image_size = 32
nc = 3  # 输入通道数（RGB）

# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载 CIFAR-10 数据集
train_dataset = torchvision.datasets.CIFAR10(root='E:/Data',
                                             train=True,
                                             transform=transform,
                                             download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = image_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(nz + num_classes, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 8x8 -> 16x16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # 16x16 -> 32x32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(True),
            nn.Conv2d(64, nc, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # 将噪声和标签连接
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Conv2d(nc + num_classes, 64, 3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * (image_size // 4) ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # 获取标签嵌入并扩展为图像大小
        label_emb = self.label_embedding(labels)
        label_emb = label_emb.unsqueeze(2).unsqueeze(3)
        label_emb = label_emb.repeat(1, 1, image_size, image_size)

        # 将图像和标签嵌入连接
        d_in = torch.cat((img, label_emb), 1)
        validity = self.model(d_in)
        return validity

# 初始化模型
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 定义损失函数和优化器
adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# 训练过程
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        batch_size_i = imgs.size(0)

        # 真实图像
        real_imgs = imgs.to(device)
        labels = labels.to(device)
        real = torch.ones(batch_size_i, 1).to(device)
        fake = torch.zeros(batch_size_i, 1).to(device)

        # -----------------
        #  训练生成器
        # -----------------
        optimizer_G.zero_grad()

        # 噪声和标签
        z = torch.randn(batch_size_i, nz).to(device)
        gen_labels = torch.randint(0, num_classes, (batch_size_i,)).to(device)

        # 生成图像
        gen_imgs = generator(z, gen_labels)

        # 计算生成器损失
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, real)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  训练判别器
        # ---------------------
        optimizer_D.zero_grad()

        # 真实图像损失
        real_pred = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(real_pred, real)

        # 生成图像损失
        fake_pred = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # 总的判别器损失
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # 打印训练信息
        if i % 100 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(train_loader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    # 每个 epoch 结束后生成一些图像
    generator.eval()
    with torch.no_grad():
        sample_z = torch.randn(10, nz).to(device)
        sample_labels = torch.arange(0, 10).to(device)
        gen_imgs = generator(sample_z, sample_labels)
        gen_imgs = gen_imgs * 0.5 + 0.5  # 反归一化

        grid = torchvision.utils.make_grid(gen_imgs.cpu(), nrow=5)
        npimg = grid.numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(f'Epoch {epoch+1}')
        plt.show()

    generator.train()
