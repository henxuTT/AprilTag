import os

import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from network.decoder_net import DecoderNet

# 加载模型

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DecoderNet()
model.load_state_dict(torch.load('model_parameters.pth'))
model = model.to(device)
model.eval()  # 设置为评估模式

# 加载和预处理图像
image_path = 'dataset/tag36h11_005_img/'
# image_path1 = 'dataset/tag_05_image/001.png'
files = os.listdir(image_path)
for file in files:
    file_path = os.path.join(image_path, file)
    image = cv2.imread(file_path)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
# print(cv2.imread(image_path1).shape)

    # image = cv2.imread(image_path)
    # image = cv2.resize(image, (1080, 1080))
    h, w = image.shape[:2]  # Get the height and width of the image

    transform = ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add a batch dimension and convert to Tensor

    # Perform inference with model
    with torch.no_grad():
        _, _, _, location = model(image_tensor)

    pixel_coords = location[0].int()  # Use the first batch item and convert to int

    # Draw the predicted keypoints onto the image
    for i in range(pixel_coords.shape[0]):
        cv2.circle(image, (pixel_coords[i, 0].item(), pixel_coords[i, 1].item()), 3, (0, 0, 255), -1)

    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()