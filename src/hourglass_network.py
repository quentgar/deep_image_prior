import torch
import torch.nn as nn

""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
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

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, filter_size_down):
        super().__init__()
        
        to_pad = int((filter_size_down-1)/2)

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=filter_size_down, padding=to_pad, stride=2, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = conv_block(out_c, out_c, filter_size_down)

    def forward(self, inputs):
      
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)

        return x

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, filter_size_up, up_sampling_mode='bilinear',need1x1_up=True):
        super().__init__()
        
        self.need = need1x1_up

        self.bn1 = nn.BatchNorm2d(in_c) # Possible erreur lien entre encoder decoder et skip

        self.conv1 = conv_block(in_c,out_c,filter_size_up)
        
        if need1x1_up:
            self.conv2 = conv_block(out_c, out_c, 1)

        self.up = nn.Upsample(scale_factor=2, mode=up_sampling_mode)

    def forward(self, inputs, skip):

        x = torch.cat([inputs, skip], axis=1)
        x = self.up(x)
        x = self.bn1(x)
        x = self.conv1(x)
        if self.need:
            x = self.conv2(x)

        return x

class decoder_noskip_block(nn.Module):
    def __init__(self, in_c, out_c, filter_size_up, up_sampling_mode='bilinear', need1x1_up=True):
        super().__init__()
        
        self.need = need1x1_up

        self.bn1 = nn.BatchNorm2d(in_c) # Possible erreur lien entre encoder decoder et skip

        self.conv1 = conv_block(in_c,out_c,filter_size_up)
        
        if need1x1_up:
            self.conv2 = conv_block(out_c, out_c, 1)

        self.up = nn.Upsample(scale_factor=2, mode=up_sampling_mode)

    def forward(self, inputs):

        x = self.up(inputs)
        x = self.bn1(x)
        x = self.conv1(x)
        if self.need:
            x = self.conv2(x)

        return x

class build_hourglass(nn.Module):
    
    def __init__(self,input_depth=32,output_depth=3,
                 num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
                 num_channels_skip=[4, 4, 4, 4, 4], filter_size_down=3, filter_size_up=3, filter_skip_size=1,
                 num_scales=5, up_samp_mode='bilinear', need1x1_up=True, need_sigmoid=True):
        super().__init__()

        num_channels_down = [num_channels_down]*num_scales if isinstance(num_channels_down, int) else num_channels_down
        num_channels_up =   [num_channels_up]*num_scales if isinstance(num_channels_up, int) else num_channels_up
        num_channels_skip = [num_channels_skip]*num_scales if isinstance(num_channels_skip, int) else num_channels_skip

        self.num_channels_skip = num_channels_skip
        self.need_sigmoid = need_sigmoid

        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

        self.num_scales = num_scales 
        
        attributes = []
        for i in range(num_scales):

          """ Encoder et Skip"""
          if i == 0:
            attributes.append(('e'+str(i+1),encoder_block(input_depth, num_channels_down[0],filter_size_down).type(torch.cuda.FloatTensor)))
            if num_channels_skip[i] != 0: # Ne pas créer de bloc skip s'il n'en existe pas
              attributes.append(('s'+str(i+1),conv_block(num_channels_down[0], num_channels_skip[i], filter_skip_size).type(torch.cuda.FloatTensor)))
          else:
            attributes.append(('e'+str(i+1),encoder_block(num_channels_down[i-1], num_channels_down[i], filter_size_down).type(torch.cuda.FloatTensor)))
            if num_channels_skip[i] != 0:
              attributes.append(('s'+str(i+1),conv_block(num_channels_down[i], num_channels_skip[i], filter_skip_size).type(torch.cuda.FloatTensor)))

          """ Decoder """
          if i == (num_scales-1):
            "Fond du réseau"
            if num_channels_skip[i] != 0:
              attributes.append(('d'+str(i+1),decoder_block(num_channels_down[i]+num_channels_skip[i], num_channels_up[i], filter_size_up, up_sampling_mode=up_samp_mode, need1x1_up=need1x1_up).type(torch.cuda.FloatTensor)))
            else: # Pas de skip
              attributes.append(('d'+str(i+1),decoder_noskip_block(num_channels_down[i]+num_channels_skip[i], num_channels_up[i], filter_size_up, up_sampling_mode=up_samp_mode, need1x1_up=need1x1_up).type(torch.cuda.FloatTensor)))
          else:
            if num_channels_skip[i] != 0:
              attributes.append(('d'+str(i+1),decoder_block(num_channels_up[i+1]+num_channels_skip[i], num_channels_up[i], filter_size_up, up_sampling_mode=up_samp_mode, need1x1_up=need1x1_up).type(torch.cuda.FloatTensor)))
            else:
              attributes.append(('d'+str(i+1),decoder_noskip_block(num_channels_up[i+1]+num_channels_skip[i], num_channels_up[i], filter_size_up, up_sampling_mode=up_samp_mode, need1x1_up=need1x1_up).type(torch.cuda.FloatTensor)))


        for key, value in attributes:
          setattr(self, key, value)

        self.conv = nn.Conv2d(num_channels_up[0],output_depth,1,padding=0, padding_mode='reflect')
        self.act = nn.Sigmoid()


    def forward(self, inputs):

        encoder = []
        for i in range(self.num_scales):
          if i == 0:
            encoder.append(getattr(self, 'e'+str(i+1))(inputs))
          else:
            encoder.append(getattr(self, 'e'+str(i+1))(encoder[i-1]))

        skip = []
        for i in range(self.num_scales):
            if self.num_channels_skip[i] != 0:
              skip.append(getattr(self, 's'+str(i+1))(encoder[i]))
            else:
              skip.append(1)

        decoder = []
        for i in range(self.num_scales,0,-1):
          if i == self.num_scales:
            if self.num_channels_skip[i-1] != 0:
              decoder.append(getattr(self, 'd'+str(i))(encoder[-1],skip[-1]))
            else:
              decoder.append(getattr(self, 'd'+str(i))(encoder[-1]))
          else:
            if self.num_channels_skip[i-1] != 0:
              decoder.append(getattr(self, 'd'+str(i))(decoder[self.num_scales-i-1],skip[i-1]))
            else:
              decoder.append(getattr(self, 'd'+str(i))(decoder[self.num_scales-i-1]))
        
        c = self.conv(decoder[-1])
        if self.need_sigmoid:
            output = self.act(c)
        else:
            output = c

        return output
