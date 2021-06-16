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
    
class build_hourglass(nn.Module):
    
    def __init__(self,input_depth=32,output_depth=3,
                 num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
                 num_channels_skip=[4, 4, 4, 4, 4], n_scales=5):
        super().__init__()
        
        attributes = []
        test = ['t1', 't2', 't3', 't4', 't5']
        for i in range(n_scales):

          """ Encoder """
          if i == 0:
            attributes.append(('e'+str(i+1),encoder_block(input_depth, num_channels_down[0]).type(torch.cuda.FloatTensor)))
            attributes.append(('s'+str(i+1),conv_block(num_channels_down[0], num_channels_skip[i], 1).type(torch.cuda.FloatTensor)))
          else:
            attributes.append(('e'+str(i+1),encoder_block(num_channels_down[i-1], num_channels_down[i]).type(torch.cuda.FloatTensor)))
            attributes.append(('s'+str(i+1),conv_block(num_channels_down[i-1], num_channels_skip[i], 1).type(torch.cuda.FloatTensor)))

          if i == n_scales-1:
            attributes.append(('d'+str(i+1),decoder_block(num_channels_up[i], num_channels_up[i]).type(torch.cuda.FloatTensor)))
          else:
            attributes.append(('d'+str(i+1),decoder_block(num_channels_up[i], num_channels_up[i+1]).type(torch.cuda.FloatTensor)))


        for key, value in attributes:
          setattr(self, key, value)

        """
        self.e1 = encoder_block(32, 128)
        self.e2 = encoder_block(128, 128)
        self.e3 = encoder_block(128, 128)
        self.e4 = encoder_block(128, 128)
        self.e5 = encoder_block(128, 128)


        self.d1 = decoder_block(128, 128)
        self.d2 = decoder_block(128, 128)
        self.d3 = decoder_block(128, 128)
        self.d4 = decoder_block(128, 128)
        self.d5 = decoder_block(128, 128)


        self.s1 = conv_block(128,4,1)
        self.s2 = conv_block(128,4,1)
        self.s3 = conv_block(128,4,1)
        self.s4 = conv_block(128,4,1)
        self.s5 = conv_block(128,4,1)"""

        self.conv = nn.Conv2d(num_channels_up[-1],output_depth,1,padding=0, padding_mode='reflect')
        self.act = nn.Sigmoid()


    def forward(self, inputs):
        """ Encoder """
        e1 = self.e1(inputs)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)

        """ Skip """
        s1 = self.s1(e1)
        s2 = self.s2(e2)
        s3 = self.s3(e3)
        s4 = self.s4(e4)
        s5 = self.s5(e5)

        """ Decoder """
        d1 = self.d1(e5, s5)
        d2 = self.d2(d1, s4)
        d3 = self.d3(d2, s3)
        d4 = self.d4(d3, s2)
        d5 = self.d5(d4, s1)

        c = self.conv(d5)
        output = self.act(c)

        return output
