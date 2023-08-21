import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


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

        # Regressor for the affine transformation
        self.fc_loc = nn.Sequential(
            nn.Linear(36000, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 36000)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        # print("theta shape", theta.shape)
        # print("x shape", x.shape)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


class AprilTagFeatureNet(nn.Module):
    def __init__(self):
        super(AprilTagFeatureNet, self).__init__()
        self.stn = STN()

        # Load the MobileNetV2 model
        self.mobilenet = mobilenet_v2(pretrained=False).features
        # Change the first convolution layer to accept grayscale images
        self.mobilenet[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # Define the regressor for 4 corners
        self.regressor = nn.Linear(1280, 66)  # 1280 is the output feature size of MobileNetV2

    def forward(self, x):
        # Transform the input
        x = self.stn(x)

        # Extract features with MobileNet
        x = self.mobilenet(x)
        x = x.mean([2, 3])

        # Predict the coordinates
        coords = self.regressor(x).view(-1, 33, 2)

        return coords