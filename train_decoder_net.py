import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import os
import numpy as np
import cv2
from network.decoder_net import DecoderNet
from torchvision.ops import nms


# 自定义数据集类
class KeypointDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]

        # 读取图像
        image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image)

        # 读取标签
        with open(label_path, 'r') as f:
            lines = f.readlines()

        keypoints = []
        for line in lines:
            x, y = line.strip().split()
            keypoints.append([float(x), float(y)])

        # 转换为Tensor
        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        return image, keypoints


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# 定义图像路径和标签路径
image_paths = './dataset/tag36h11_005_img/'  # 图像文件路径
file_list = os.listdir(image_paths)  # 获取文件夹中所有文件的名称
image_list = [os.path.join(image_paths, filename) for filename in file_list]
input_img_shape = cv2.imread(image_list[0]).shape[:2]
label_list = [file_name.replace('.jpg', '.txt').replace('img', 'label') for file_name in image_list]


model = DecoderNet().to(device)


# 创建数据集实例
dataset = KeypointDataset(image_list, label_list, transform=ToTensor())

# 创建数据加载器
batch_size = 8
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 进行训练
num_epochs = 10
i=1

for epoch in range(num_epochs):
    for images, keypoints in data_loader:
        images = images.to(device)
        keypoints = keypoints.to(device)

        # 前向传播
        init_masks, corners, confidence, location = model(images)

        # location = location.permute(0, 2, 3, 1).contiguous().view(location.size(0), -1, 2)

        # 计算损失
        loss = criterion(location, keypoints)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(i, '  batch ', loss)
        i = i + 1

        # 打印每个epoch的损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

# 将模型转回CPU后再保存参数
model.to('cpu')
torch.save(model.state_dict(), 'model_parameters.pth')


