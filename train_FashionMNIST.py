# 导入相关包
import torch
from torch import optim
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import math

import dcgan
import gan

# 超参数
batch_size = 128
epochs = 200
learning_rate = 1e-4

# 加载数据集
# 预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = torchvision.datasets.FashionMNIST('data',
                                    train=True,
                                    transform=transform,
                                    download=True)

dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# 可视化部分数据
plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(transforms.ToPILImage()(dataset[i][0]), cmap="gray")
    plt.axis('off')
plt.savefig(f"./view_data.png")
plt.close()

# 训练设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 构造网络
generator = dcgan.GeneratorMNIST().to(device)
discriminator = dcgan.DiscriminatorMNIST().to(device)

# 优化器
g_optim = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
d_optim = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# 损失函数
criterion = nn.BCELoss().to(device)

# 训练
# 随机生成16batch,100长的噪声作为Generator的输入
test_input = torch.randn(16, 100).to(device)

# 损失函数记录
D_loss = []
G_loss = []
iterations = []

iteration = 0
# 开始训练
for epoch in range(epochs):
    for step, (imgs, tragets) in enumerate(dataLoader):
        print(f"\r进度:{step + 1} / {math.ceil(len(dataset) / batch_size)}", end="")
        imgs = imgs.to(device)
        random_noise = torch.randn(batch_size, 100).to(device)
        
        # 在判别器训练真实图片
        d_optim.zero_grad()
        # 判别器输入真实的图片, 希望real_output是1
        real_output = discriminator(imgs)
        # 得到判别器在真实数据上的损失
        d_real_loss = criterion(real_output, torch.ones_like(real_output))
        d_real_loss.backward()

        # 在判别器训练生成的假图片
        gen_img = generator(random_noise)
        # 此时优化的对象是判别器, 要把生成器的梯度截断
        # 判别器输入生成的图片, fake_output对生成的图片预测
        fake_output = discriminator(gen_img.detach())
         # 得到判别器在生成数据上的损失
        d_fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
        d_fake_loss.backward()
        d_loss = d_real_loss.item() + d_fake_loss.item()
        d_optim.step()

        # 对生成器进行优化
        g_optim.zero_grad()
        # 希望其骗过判别器
        fake_output = discriminator(gen_img)
        # 得到生成器的损失
        g_loss = criterion(fake_output, torch.ones_like(fake_output))  
        g_loss.backward()
        g_optim.step()
        
        # 记录损失函数
        if iteration == 1 or iteration % 100 == 0:
            D_loss.append(d_loss)
            G_loss.append(g_loss.item())
            iterations.append(iteration)
        iteration = iteration + 1
    
    # 输出信息
    print(f' 结果：Epoch [{epoch+1}/{epochs}] Loss D: {d_loss}, Loss G: {g_loss.item()}')
    # 保存模型
    state = {
        'epoch' : epoch + 1,
        'd_loss' : d_loss,
        'g_loss' : g_loss.item(),
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, 'checkpoint/ckpt.pth')
    # 记录此轮生成图像
    if (epoch + 1) % 10 == 0 or epoch == 0:
        predict = np.squeeze(np.squeeze(generator(test_input).detach().cpu().numpy()))
        plt.figure(figsize=(4, 4))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            # 先归一化，再转换格式
            plt.imshow((predict[i] + 1) / 2, cmap="gray")
            plt.axis('off')
        plt.savefig(f"./generated_img/generated_image_epoch_{epoch+1}.jpg")
        plt.close()

# 展示损失函数曲线
plt.figure(figsize=(4, 4))
plt.title("loss")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.plot(iterations, G_loss, c="tab:red", label='G_loss')
plt.plot(iterations, D_loss, c="tab:blue", label='D_loss')
plt.legend()
plt.savefig(f"./loss.jpg")
plt.close()