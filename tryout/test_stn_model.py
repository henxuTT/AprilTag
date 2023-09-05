import torch
from torch import nn
from torchvision.models import mobilenet_v2
from network.stn_net import STN
import os
import cv2
import numpy as np
from torchvision.transforms import ToTensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = STN()  # 创建一个模型的实例
model.load_state_dict(torch.load('../model_parameters.pth'))  # 加载参数
model = model.to(device)  # 移动模型到正确的设备上
model.eval()  # 将模型设置为评估模式

image_path = '../dataset/STN_dataset/tag36h11_005_img/train/'
files = os.listdir(image_path)
for file in files:
    file_path = os.path.join(image_path, file)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    transform = ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)

    output = model(image_tensor)
    tensor = output.view(3, 3)
    numpy_array = tensor.cpu().detach().numpy()  # 移动到CPU并转换为Numpy数组
    print(numpy_array)
