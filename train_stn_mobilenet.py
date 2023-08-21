import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from network.stn_mobilenet import AprilTagFeatureNet
import os
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

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
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # print(image.shape)
        # image = cv2.Canny(image, 100, 200)  # 这里的 100 和 200 是阈值，你可以根据需要调整
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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path = 'dataset/stn_mobilenet/tag36h11_005_img/train/'  # 图像文件路径
file_list = os.listdir(image_path)  # 获取文件夹中所有文件的名称
image_list = [os.path.join(image_path, filename) for filename in file_list]
label_list = [file_name.replace('.jpg', '.txt').replace('img', 'label') for file_name in image_list]

model = AprilTagFeatureNet().to(device)
dataset = KeypointDataset(image_list, label_list, transform=ToTensor())
batch_size = 8
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
i=1
for epoch in range(num_epochs):
    for images, true_coords in data_loader:
        images = images.to(device)
        true_coords = true_coords.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        predicted_coords = model(images)

        # Calculate loss
        loss = criterion(predicted_coords, true_coords)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        print(i, '  batch ', loss)
        i = i + 1

    # # Validation loop
    # model.eval()  # Set the model to evaluation mode
    # val_loss = 0.0
    # with torch.no_grad():
    #     for images, true_coords in val_loader:
    #         images = images.to(device)
    #         true_coords = true_coords.to(device)
    #
    #         # Forward pass
    #         predicted_coords = model(images)
    #
    #         # Calculate loss
    #         loss = criterion(predicted_coords, true_coords)
    #         val_loss += loss.item()
    #
    # val_loss /= len(val_loader)
    #
    # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss}")
# 将模型转回CPU后再保存参数
model.to('cpu')
torch.save(model.state_dict(), 'stn_mobilenet_paras.pth')
