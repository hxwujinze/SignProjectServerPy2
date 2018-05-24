import torch
import torch.nn as nn

LEARNING_RATE = 0.00018
WEIGHT_DECAY = 0.0000002
EPOCH = 1200
BATCH_SIZE = 64

class CNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # input 14 x 64
        self.conv1 = nn.Sequential(
            # 使用VGGNet架构卷积
            nn.Conv1d(
                in_channels=14,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=1,
            ),  # Lout=floor((Lin+2*padding-dilation*(kernel_size -1 ) - 1)/stride+1)
            # output 28 x 64
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            # output 28 x 64
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=3)  # 32 x 21
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),  # 32 x 21
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),  # 32 x 21
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),  # 32 x 21
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=2),  # 64 x 10
        )

        self.out1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(64 * 10, 512),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 24),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out1(x)
        return x

def get_max_index(tensor):
    # print('置信度')
    # print(tensor.data.float()[0])
    tensor = torch.max(tensor, dim=1)[1]
    # 对矩阵延一个固定方向取最大值
    return torch.squeeze(tensor).data.int()

def output_len(Lin, padding, kernel_size, stride):
    Lout = (Lin + 2 * padding - (kernel_size - 1) - 1) / stride + 1
    return Lout
