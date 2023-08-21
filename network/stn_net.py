# import torch
# from torch import nn
# from torchvision.models import mobilenet_v2
# from torch.nn import functional as F
# from network.conv import conv, conv_dw, conv_dw_no_bn
#
#
# INTER_CHANNEL_NUM = 512
# INTER_CHANNEL_NUM_BLOCK = 512
# INTER_CHANNEL_NUM_REFINE =128
#
#
# class STN(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(STN, self).__init__()
#
#         # Localization Network.
#         self.localization = nn.Sequential(
#             nn.Conv2d(in_channels, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True)
#         )
#
#         # Fully connected layer for transformation matrix prediction.
#         self.fc_loc = nn.Sequential(
#             nn.Linear(10 * 3 * 3, 32),
#             nn.ReLU(True),
#             nn.Linear(32, 3 * 3)
#         )
#
#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float))
#
#     # STN forward function
#     def forward(self, x):
#         xs = self.localization(x)
#         xs = xs.view(-1, 10 * 3 * 3)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 3, 3)
#
#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)
#
#         return x
#
#
#
#
# class DecoderNet(nn.Module):
#     def __init__(self, num_channels=32, num_masks=2, num_classes=3):
#         super().__init__()
#         self.model = nn.Sequential(
#             conv(3, INTER_CHANNEL_NUM // 16, stride=2, bias=False),
#             conv_dw(INTER_CHANNEL_NUM // 16, INTER_CHANNEL_NUM // 8),
#             conv_dw(INTER_CHANNEL_NUM // 8, INTER_CHANNEL_NUM // 4, stride=2),
#             conv_dw(INTER_CHANNEL_NUM // 4, INTER_CHANNEL_NUM // 4),
#             conv_dw(INTER_CHANNEL_NUM // 4, INTER_CHANNEL_NUM // 2, stride=2),
#             conv_dw(INTER_CHANNEL_NUM // 2, INTER_CHANNEL_NUM // 2),
#             conv_dw(INTER_CHANNEL_NUM // 2, INTER_CHANNEL_NUM),  # conv4_2
#             conv_dw(INTER_CHANNEL_NUM, INTER_CHANNEL_NUM, dilation=2, padding=2),
#             conv_dw(INTER_CHANNEL_NUM, INTER_CHANNEL_NUM),
#             conv_dw(INTER_CHANNEL_NUM, INTER_CHANNEL_NUM),
#             conv_dw(INTER_CHANNEL_NUM, INTER_CHANNEL_NUM),
#             conv_dw(INTER_CHANNEL_NUM, INTER_CHANNEL_NUM)  # conv5_5
#         )
#         self.cpm = Cpm(INTER_CHANNEL_NUM, num_channels)
#         self.initial_stage = InitialStage(num_channels, num_masks)
#         self.align = conv(num_channels + num_masks, num_channels, kernel_size=1, padding=0, bn=False)
#         self.conner_regressor = CornerRegressor(num_channels)
#
#         self.keypoint_regressor = KeypointRegressor(num_channels=32, num_classes=3, num_keypoints=33)
#
#     def forward(self, x):
#         model_features = self.model(x)
#         backbone_features = self.cpm(model_features)
#         init_masks = self.initial_stage(backbone_features)[0]
#         x1 = self.align(torch.cat([init_masks, backbone_features], dim=1))
#         corners = self.conner_regressor(x1)
#         confidence, location = self.keypoint_regressor(x1)
#         return init_masks, corners, confidence, location

import torch
from torch import nn
import torch.nn.functional as F

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 60 * 60, 640),
            nn.ReLU(True),
            nn.Linear(640, 8)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float))

    # Spatial transformer network forward function

    @staticmethod
    def apply_projection(x, theta):
        N, C, H, W = x.size()

        # 创建规范化网格
        x_coords, y_coords = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
        x_coords = x_coords.to(x.device).view(-1)
        y_coords = y_coords.to(x.device).view(-1)
        ones = torch.ones_like(x_coords)
        grid_norm = torch.stack([x_coords, y_coords, ones], dim=1)  # 形状 [H * W, 3]
        grid_norm = grid_norm[None, :, :]  # 增加一个批量维度，形状 [1, H * W, 3]
        grid_norm = grid_norm.repeat(N, 1, 1)  # 重复批量维度以匹配 theta 的形状

        # 应用投影变换
        grid_projected = torch.bmm(grid_norm, theta.transpose(1, 2))  # 形状 [N, H * W, 3]
        x_projected = grid_projected[..., 0] / (grid_projected[..., 2] + 1e-7)
        y_projected = grid_projected[..., 1] / (grid_projected[..., 2] + 1e-7)

        # 重新调整形状
        x_projected = x_projected.view(N, H, W)
        y_projected = y_projected.view(N, H, W)

        # 堆叠 x 和 y
        grid_final = torch.stack([x_projected, y_projected], dim=3)  # 形状 [N, H, W, 2]

        return F.grid_sample(x, grid_final)

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 60 * 60)
        theta = self.fc_loc(xs)
        batch_size, _ = theta.size()  # 获取批量大小
        ones_column = torch.ones(batch_size, 1, device=theta.device)  # 创建一个与批量大小匹配的全1列
        theta = torch.cat((theta, ones_column), dim=1)  # 沿着列方向拼接

        theta = theta.view(-1, 3, 3)

        x_projected = STN.apply_projection(x, theta)

        return x_projected, theta


