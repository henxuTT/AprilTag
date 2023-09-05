import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor

from network.stn_net_2 import STN
import os
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
# import pytorch_ssim


import torch
import torch.nn.functional as F


def ssim(img1, img2, window_size=5, C1=0.01 ** 2, C2=0.03 ** 2):
    window = torch.ones((1, 1, window_size, window_size)).to(img1.device)
    window /= window.sum()

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=1)

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=1) - mu1 * mu1
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=1) - mu2 * mu2
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=1) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=5, C1=0.01 ** 2, C2=0.03 ** 2):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2

    def forward(self, img1, img2):
        # 这里可以使用上一个答案中的ssim函数
        ssim_val = ssim(img1, img2, self.window_size, self.C1, self.C2)
        return 1 - ssim_val


class MatrixDataset(Dataset):
    def __init__(self, image_paths, label_path, transform=None):
        self.image_paths = image_paths
        self.label_path = label_path
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        # label_path = self.label_paths[index]

        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        # image = cv2.Canny(image, 100, 200)  # 这里的 100 和 200 是阈值，你可以根据需要调整
        if self.transform:
            image = self.transform(image)

        label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_image = cv2.resize(label_image, (256, 256))
        if self.transform:
            label_image = self.transform(label_image)
            # print(label_image)
        # # 读取标签
        # matrix = np.loadtxt(label_path)
        #
        # # 转换为Tensor
        # matrix = torch.tensor(matrix, dtype=torch.float32).flatten()
        return image, label_image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_path = '../dataset/STN_dataset/tag36h11_005_img/train/'  # 图像文件路径
file_list = os.listdir(image_path)  # 获取文件夹中所有文件的名称
image_list = [os.path.join(image_path, filename) for filename in file_list]

# label_list = [file_name.replace('.jpg', '.txt').replace('img', 'label') for file_name in image_list]
label_path = '../images/tag36h11_005_000.jpg'
dataset = MatrixDataset(image_list, label_path, transform=ToTensor())
batch_size = 8
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create the network
model = STN().to(device)

# Choose a loss function and an optimizer
criterion = nn.KLDivLoss(log_target=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# The number of epochs is the number of times you go through the full dataset
num_epochs = 10
i = 1

for epoch in range(num_epochs):
    for images, label_image in data_loader:
        images = images.to(device)
        label_image = label_image.to(device)

        label_trained, _ = model(images)
        loss = criterion(label_trained, label_image)

        optimizer.zero_grad()
        loss.backward()  # backpropagation
        optimizer.step()  # update the weights
        print(i, '  batch ', loss)
        i = i + 1

        # 打印每个epoch的损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

# 将模型转回CPU后再保存参数
model.to('cpu')
torch.save(model.state_dict(), '../model_parameters.pth')

