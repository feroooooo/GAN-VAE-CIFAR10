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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging

import gan

# 超参数
batch_size = 64
epoch_num = 200
learning_rate = 2e-4
beta1 = 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path = 'E:/Data'

# 初始化Tensorboard
writer = SummaryWriter()
os.mkdir(f'{writer.log_dir}/checkpoints')
os.mkdir(f'{writer.log_dir}/generated_img')

# 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 设置日期格式
    encoding='utf-8',
    filename=f'{writer.log_dir}/train.log',  # 输出日志到文件
    filemode='w'  # 写入模式为覆盖 (可选 'a' 为追加)
)

logging.info(f"Current Device: {device}")
logging.info(f"Epoch Num: {epoch_num}")
logging.info(f"Batch Size: {batch_size}")
logging.info(f"Learning Rate: {learning_rate}")


# 加载数据集
# 预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 数据集的均值
                         (0.2023, 0.1994, 0.2010))  # CIFAR-10 数据集的标准差
])
dataset = torchvision.datasets.CIFAR10(data_path,
                                    train=True,
                                    transform=transform,
                                    download=True)

dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 可视化部分数据
plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    # 反归一化图像数据
    img = dataset[i][0]
    img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    # 将值裁剪到[0,1]范围内
    img = torch.clamp(img, 0, 1)
    plt.imshow(transforms.ToPILImage()(img))
    plt.axis('off')
plt.savefig(f"{writer.log_dir}/view_data.png")
plt.close()

# 构造网络
generator = gan.Generator().to(device)
discriminator = gan.Discriminator().to(device)

# 优化器
g_optim = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
d_optim = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# 损失函数
criterion = nn.BCELoss().to(device)

# 训练
# 随机生成16batch,100长的噪声作为Generator的输入
test_input = torch.randn(16, 100).to(device)

# 损失函数记录

# 开始训练
for epoch in range(epoch_num):
    generator.train()
    discriminator.train()
    g_loss_total = 0
    d_loss_total = 0
    for imgs, tragets in tqdm(dataLoader):
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
        d_loss_total += d_loss
        d_optim.step()

        # 对生成器进行优化
        g_optim.zero_grad()
        # 希望其骗过判别器
        fake_output = discriminator(gen_img)
        # 得到生成器的损失
        g_loss = criterion(fake_output, torch.ones_like(fake_output))  
        g_loss.backward()
        g_loss_total += g_loss.item()
        g_optim.step()
        
    
    g_loss_avg = g_loss_total / len(dataLoader)
    d_loss_avg = d_loss_total / len(dataLoader)
    # 输出信息
    print(f'Epoch: {epoch + 1}/{epoch_num} | G Loss: {g_loss_avg:.4f} | D Loss: {d_loss_avg:.4f}')
    logging.info(f'Epoch: {epoch + 1}/{epoch_num} | G Loss: {g_loss_avg:.4f} | D Loss: {d_loss_avg:.4f}')
    writer.add_scalars('Loss', {'G': g_loss_avg, 'D': d_loss_avg}, global_step=epoch + 1)
    

    # 每10轮记录一次生成图像
    if (epoch + 1) % 10 == 0 or epoch == 0:
        generator.eval()
        with torch.no_grad():
            predict = generator(test_input).detach().cpu().numpy()
            predict = np.transpose(predict, (0, 2, 3, 1))  # 转换为(N,H,W,C)格式
            plt.figure(figsize=(4, 4))
            for i in range(16):
                plt.subplot(4, 4, i + 1)
                # 生成器输出范围是(-1,1),需要转换到(0,1)
                img = (predict[i] + 1) / 2
                # 确保像素值在合理范围内
                img = np.clip(img, 0, 1)
                plt.imshow(img)
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{writer.log_dir}/generated_img/generated_image_epoch_{epoch+1}.png")
            plt.close()
            # 保存模型
            state = {
                'epoch' : epoch + 1,
                'd_loss' : d_loss_avg,
                'g_loss' : g_loss_avg,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
            }
            torch.save(state, f'{writer.log_dir}/checkpoints/ckpt_{epoch+1}.pth')
