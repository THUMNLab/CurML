import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, (3,3), padding=0, stride=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (3,3), padding=0, stride=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 32, (3,3), padding=0)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, (3,3), padding=0)
        self.batchnorm4 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2,2)

        self.layer1 = nn.Linear(32 * 5 * 5, num_classes)
        self.batchnorm = nn.BatchNorm1d(num_classes)
        self.relu = nn.ReLU()

    
    def forward(self,x):
        out = self.relu(self.batchnorm1(self.conv1(x)))
        out = self.relu(self.batchnorm2(self.conv2(out)))
        out = self.pool1(out)
        out = self.relu(self.batchnorm3(self.conv3(out)))
        out = self.relu(self.batchnorm4(self.conv4(out)))
        out = self.pool2(out)
        out = out.view(-1, 32 * 5 * 5)
        out = self.batchnorm(self.layer1(out))

        return out