import os
import math

import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torch
from torch import optim
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import dcgan
import gan


test_input = torch.randn(2, 100)
Gen = dcgan.GeneratorMNIST()
Dis = dcgan.DiscriminatorCorn()
image = torch.randn(1, 3, 64, 64)
print(Dis(image).shape)