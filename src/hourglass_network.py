import torch
import torch.nn as nn

""" Convolutional block """
class conv_block(nn.Module):
    def __init__(self, in_c, out_c, k_size):
        super().__init__()

        to_pad = int((k_size-1)/2)

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=k_size, padding=to_pad, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        return x
      
""" Encoder block """
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=2, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = conv_block(out_c, out_c, 3)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)

        return x
      
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_c+4) # Possible erreur lien entre encoder decoder et skip

        self.conv1 = conv_block(in_c+4,out_c,3)

        self.conv2 = conv_block(out_c, out_c, 1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, inputs, skip):
        x = torch.cat([inputs, skip], axis=1)
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.up(x)

        return x
