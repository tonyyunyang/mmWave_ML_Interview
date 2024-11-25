import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(InceptionModule, self).__init__()
        
        # 1x1 conv branch
        self.conv1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_1x1)
        
        # 1x1 conv -> 3x3 conv branch
        self.conv3_reduce = nn.Conv2d(in_channels, red_3x3, kernel_size=1)
        self.bn3_reduce = nn.BatchNorm2d(red_3x3)
        self.conv3 = nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_3x3)
        
        # 1x1 conv -> 5x5 conv branch
        self.conv5_reduce = nn.Conv2d(in_channels, red_5x5, kernel_size=1)
        self.bn5_reduce = nn.BatchNorm2d(red_5x5)
        self.conv5 = nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm2d(out_5x5)
        
        # max pool -> 1x1 conv branch
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_proj = nn.Conv2d(in_channels, out_pool, kernel_size=1)
        self.bn_pool = nn.BatchNorm2d(out_pool)

    def forward(self, x):
        # 1x1 conv
        conv1 = F.relu(self.bn1(self.conv1(x)))
        
        # 1x1 conv -> 3x3 conv
        conv3 = F.relu(self.bn3_reduce(self.conv3_reduce(x)))
        conv3 = F.relu(self.bn3(self.conv3(conv3)))
        
        # 1x1 conv -> 5x5 conv
        conv5 = F.relu(self.bn5_reduce(self.conv5_reduce(x)))
        conv5 = F.relu(self.bn5(self.conv5(conv5)))
        
        # max pool -> 1x1 conv
        pool = self.pool(x)
        pool = F.relu(self.bn_pool(self.pool_proj(pool)))
        
        # Concatenate along channel dimension
        return torch.cat([conv1, conv3, conv5, pool], dim=1)

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        
        self.config = config
        in_channels = config['in_channels']
        class_num = config['class_num']
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Inception modules
        self.inception1 = InceptionModule(64, 64, 96, 128, 16, 32, 32)  # out: 256 channels
        self.inception2 = InceptionModule(256, 128, 128, 192, 32, 96, 64)  # out: 480 channels
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(480, class_num)

    def forward(self, x):
        # Initial convolution and pooling
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        
        # Inception modules
        x = self.inception1(x)
        x = self.inception2(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Classification layer
        x = self.dropout(x)
        x = self.fc(x)
        
        return x