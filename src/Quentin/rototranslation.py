import torch
import numpy as np
from RotoTrans import rotation
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter

from src.utils import *
from src.hourglass_network import *
import math as m

dtype = torch.cuda.FloatTensor

def rotate_lifting_kernels(kernel, orientations_nb, periodicity=2 * np.pi, diskMask=True):
    """ Rotates the set of 2D lifting kernels.
        INPUT:
            - kernel, a tensor flow tensor with expected shape:
                [ChannelsOUT, ChannelsIN, Height, Width]
            - orientations_nb, an integer specifying the number of rotations
        INPUT (optional):
            - periodicity, rotate in total over 2*np.pi or np.pi
            - disk_mask, True or False, specifying whether or not to mask the kernels spatially
        OUTPUT:
            - set_of_rotated_kernels, a tensorflow tensor with dimensions:
                [nbOrientations, ChannelsOUT, ChannelsIN, Height, Width]
    """

    # Unpack the shape of the input kernel
    channelsOUT, channelsIN, kernelSizeH, kernelSizeW = map(int, kernel.shape)
    
    #print("Z2-SE2N BASE KERNEL SHAPE:", kernel.shape)  # Debug

    # Flatten the baseline kernel
    # Resulting shape: [channelsIN*channelsOUT, kernelSizeH*kernelSizeW]
    kernel_flat = torch.reshape(
        kernel, (channelsIN * channelsOUT, kernelSizeH * kernelSizeW))
    
    # Transpose: [kernelSizeH*kernelSizeW, channelsIN*channelsOUT]
    kernel_flat = torch.transpose(kernel_flat,0,1)

    #print("Flatten kernel shape : ",kernel_flat.shape)

    # Generate a set of rotated kernels via rotation matrix multiplication
    # For efficiency purpose, the rotation matrix is implemented as a sparse matrix object
    # Result: The non-zero indices and weights of the rotation matrix
    idx, vals = rotation.MultiRotationOperatorMatrixSparse(
        [kernelSizeH, kernelSizeW],
        orientations_nb,
        periodicity=periodicity,
        diskMask=diskMask)
    
    new_idx = [[],[]]
    for i in range(len(idx)):
      new_idx[0].append(idx[i][0])
      new_idx[1].append(idx[i][1])
    

    # Sparse rotation matrix
    # Resulting shape: [nbOrientations*kernelSizeH*kernelSizeW,
    # kernelSizeH*kernelSizeW]

    rotOp_matrix = torch.sparse_coo_tensor(
        new_idx, vals,
        torch.Size([orientations_nb * kernelSizeH * kernelSizeW, kernelSizeH * kernelSizeW]))

    # Matrix multiplication
    # Resulting shape: [nbOrientations*kernelSizeH*kernelSizeW,
    # channelsIN*channelsOUT]
    set_of_rotated_kernels = torch.sparse.mm(
        rotOp_matrix.type(dtype), kernel_flat.type(dtype))
    
    #print("Set of rotated kernels before reshape : ",set_of_rotated_kernels.shape)

    # Reshaping
    # Resulting shape: [nbOrientations, channelsOUT, channelsIN, kernelSizeH, kernelSizeW]
    set_of_rotated_kernels = torch.reshape(
        set_of_rotated_kernels, (orientations_nb, channelsOUT, channelsIN, kernelSizeH, kernelSizeW))
    
    #print("Set of rotated kernels after reshape : ",set_of_rotated_kernels.shape)

    return set_of_rotated_kernels

def rotate_gconv_kernels(kernel, periodicity=2 * np.pi, diskMask=True):
  """ Rotates the set of SE2 kernels. 
        Rotation of SE2 kernels involves planar rotations and a shift in orientation,
        see e.g. the left-regular representation L_g of the roto-translation group on SE(2) images,
        (Eq. 3) of the MICCAI 2018 paper.
        INPUT:
            - kernel, a tensor flow tensor with expected shape:
                [nbOrientations, ChannelsOUT, ChannelsIN, Height, Width]
        INPUT (optional):
            - periodicity, rotate in total over 2*np.pi or np.pi
            - disk_mask, True or False, specifying whether or not to mask the kernels spatially
        OUTPUT:
            - set_of_rotated_kernels, a tensorflow tensor with dimensions:
                [nbOrientations, Height, Width, ChannelsIN, ChannelsOUT, nbOrientations]
              I.e., for each rotation angle a rotated (shift-twisted) version of the input kernel.
  """

  # Rotation of an SE2 kernel consists of two parts:
  # PART 1. Planar rotation
  # PART 2. A shift in theta direction

  # Unpack the shape of the input kernel
  orientations_nb, channelsOUT, channelsIN, kernelSizeH, kernelSizeW = map(int, kernel.shape)
  
  #print("SE2N-SE2N BASE KERNEL SHAPE:", kernel.shape)  # Debug

  # PART 1 (planar rotation)
  # Flatten the baseline kernel
  # Resulting shape: [orientations_nb*channelsIN*channelsOUT, kernelSizeH*kernelSizeW]
  #
  kernel_flat = torch.reshape(kernel, [orientations_nb * channelsIN * channelsOUT, kernelSizeH * kernelSizeW])

  # Permute axis : [kernelSizeH*kernelSizeW, orientations_nb*channelsIN*channelsOUT]
  kernel_flat = torch.transpose(kernel_flat,0,1)

  #print("Flat kernel : ",kernel_flat.shape)

  # Generate a set of rotated kernels via rotation matrix multiplication
  # For efficiency purpose, the rotation matrix is implemented as a sparse matrix object
  # Result: The non-zero indices and weights of the rotation matrix
  idx, vals = rotation.MultiRotationOperatorMatrixSparse([kernelSizeH, kernelSizeW], orientations_nb, periodicity=periodicity, diskMask=diskMask)

  new_idx = [[],[]]
  for i in range(len(idx)):
    new_idx[0].append(idx[i][0])
    new_idx[1].append(idx[i][1])

  # The corresponding sparse rotation matrix
  # Resulting shape: [nbOrientations*kernelSizeH*kernelSizeW,kernelSizeH*kernelSizeW]
  #
  rotOp_matrix = torch.sparse_coo_tensor(new_idx, vals, [orientations_nb * kernelSizeH * kernelSizeW, kernelSizeH * kernelSizeW])

  # Matrix multiplication (each 2D plane is now rotated)
  # Resulting shape: [nbOrientations*kernelSizeH*kernelSizeW, orientations_nb*channelsIN*channelsOUT]
  #
  kernels_planar_rotated = torch.sparse.mm(rotOp_matrix.type(dtype), kernel_flat.type(dtype))
  
  kernels_planar_rotated = torch.reshape(kernels_planar_rotated, [orientations_nb, kernelSizeH, kernelSizeW, orientations_nb, channelsIN, channelsOUT])

  #print("Matmul reshape : ",kernels_planar_rotated.shape)

  # PART 2 (shift in theta direction)
  set_of_rotated_kernels = [None] * orientations_nb
  for orientation in range(orientations_nb):
      # [kernelSizeH,kernelSizeW,orientations_nb,channelsIN,channelsOUT]
      kernels_temp = kernels_planar_rotated[orientation]
      # [kernelSizeH,kernelSizeW,channelsIN,channelsOUT,orientations_nb]
      kernels_temp = kernels_temp.permute(0, 1, 3, 4, 2)
      # [kernelSizeH*kernelSizeW*channelsIN*channelsOUT*orientations_nb]
      kernels_temp = torch.reshape(kernels_temp, [kernelSizeH * kernelSizeW * channelsIN * channelsOUT, orientations_nb])
      # Roll along the orientation axis
      roll_matrix = torch.tensor(np.roll(np.identity(orientations_nb), orientation, axis=1)).type(dtype)
      kernels_temp = torch.matmul(kernels_temp, roll_matrix)
      kernels_temp = torch.reshape(kernels_temp, [kernelSizeH, kernelSizeW, channelsIN, channelsOUT, orientations_nb])  # [Nx,Ny,Ntheta,Nin,Nout]
      
      set_of_rotated_kernels[orientation] = kernels_temp

  stacked_kernels = torch.stack(set_of_rotated_kernels)

  #print("Stacked kernels shape : ",stacked_kernels.shape)

  return stacked_kernels

""" Constructs a group convolutional layer.
        (lifting layer from Z2 to SE2N with N input number of orientations)
        INPUT:
            - input_tensor in Z2, a tensorflow Tensor with expected shape:
                [BatchSize, ChannelsIN, Height, Width]
            - kernel, a tensorflow Tensor with expected shape:
                [ChannelsOUT, ChannelsIN, kernelSize, kernelSize]
                /!\ [] /!\
            - orientations_nb, an integer specifying the number of rotations
        INPUT (optional):
            - periodicity, rotate in total over 2*np.pi or np.pi
            - disk_mask, True or False, specifying whether or not to mask the kernels spatially
        OUTPUT:
            - output_tensor, the tensor after group convolutions with shape
                [BatchSize, orientations_nb, ChannelsOut, Height', Width']
                (Height', Width' are reduced sizes due to the valid convolution)
            - kernels_formatted, the formated kernels, i.e., the full stack of rotated kernels with shape:
                [orientations_nb, ChannelsOUT, ChannelsIN, kernelSize, kernelSize]
    """
class lifting_block(nn.Module):

  def __init__(self, channelsIN, channelsOUT, kSize, orientations_nb,
               periodicity=2 * np.pi, diskMask=True, padding='same',
               dtype = torch.cuda.FloatTensor):

    super().__init__()

    std = m.sqrt(2.0 / (channelsIN*kSize*kSize))

    self.kernel = Parameter(torch.randn((channelsOUT,channelsIN,kSize,kSize), requires_grad=True)*std)
    
    self.orientations_nb = orientations_nb
    self.channelsIN = channelsIN
    self.channelsOUT = channelsOUT
    self.kSize = kSize

    self.periodicity = periodicity
    self.diskMask = diskMask
    self.padding = padding

    self.dtype = dtype

  def forward(self, input):

    # Preparation for group convolutions
    # Precompute a rotated stack of kernels
    # Shape [nbOrientations, channelsOUT, channelsIN, kernelSizeH, kernelSizeW]
    kernel_stack = rotate_lifting_kernels(
        self.kernel, self.orientations_nb, periodicity=self.periodicity, diskMask=self.diskMask)
    
    #print("Z2-SE2N ROTATED KERNEL SET SHAPE:",kernel_stack.shape)  # Debug

    # Format the kernel stack as a 2D kernel stack (merging the rotation and
    # channelsOUT axis)
    kernels_as_if_2D = torch.reshape(
        kernel_stack, (self.orientations_nb * self.channelsOUT, 
                       self.channelsIN, self.kSize, self.kSize))
    
    #print("2D kernel stack : ",kernels_as_if_2D.shape)

    # Perform the 2D convolution

    # Input shape [1, channelsIN, h, w]
    # Kernels shape [nbOrientations*channelsOUT, channelsIN, kernelSizeH, kernelSizeW]
    layer_output = F.conv2d(input.type(self.dtype), kernels_as_if_2D.type(self.dtype),padding=self.padding)

    #print(layer_output.shape)

    # Reshape to an SE2 image (split the orientation and channelsOUT axis)
    # Note: the batch size is unknown, hence this dimension needs to be
    # obtained using the tensorflow function tf.shape, for the other
    # dimensions we keep using tensor.shape since this allows us to keep track
    # of the actual shapes (otherwise the shapes get convert to
    # "Dimensions(None)").
    layer_output = torch.reshape(
        layer_output, (layer_output.shape[0], self.orientations_nb,
                       self.channelsOUT, int(layer_output.shape[2]), int(layer_output.shape[3])))
    
    #print("OUTPUT SE2N ACTIVATIONS SHAPE:", layer_output.shape)  # Debug

    # FINAL SHAPE [1, nbOrientations, channelsOUT, h', w']

    #return layer_output, kernel_stack
    
    return layer_output

""" Constructs a group convolutional layer.
        (group convolution layer from SE2N to SE2N with N input number of orientations)
        INPUT:
            - input_tensor in SE2n, a tensor flow tensor with expected shape:
                [BatchSize, nbOrientations, ChannelsIN, Height, Width]
            - kernel, a tensorflow Tensor with expected shape:
                [nbOrientations, ChannelsOUT, ChannelsIN, kernelSize, kernelSize]
        INPUT (optional):
            - periodicity, rotate in total over 2*np.pi or np.pi
            - disk_mask, True or False, specifying whether or not to mask the
                kernels spatially
        OUTPUT:
            - output_tensor, the tensor after group convolutions with shape
                [BatchSize, nbOrientations, ChannelsOut, Height', Width']
                (Height', Width' are the reduced sizes due to the valid convolution)
            - kernels_formatted, the formated kernels, i.e., the full stack of
                rotated kernels with shape [nbOrientations, kernelSize, kernelSize, nbOrientations, channelsIn, channelsOut]
  """
class gconv_block(nn.Module):

  def __init__(self, channelsIN, channelsOUT, kSize, orientations_nb,
               periodicity=2 * np.pi, diskMask=True, padding='same',
               dtype = torch.cuda.FloatTensor):

    super().__init__()

    std = m.sqrt(2.0 / (channelsIN*kSize*kSize))

    self.kernel = Parameter(torch.randn((orientations_nb,channelsOUT,channelsIN,kSize,kSize),requires_grad=True)*std)

    self.orientations_nb = orientations_nb
    self.channelsIN = channelsIN
    self.channelsOUT = channelsOUT
    self.kSize = kSize

    self.periodicity = periodicity
    self.diskMask = diskMask
    self.padding = padding

    self.dtype = dtype

  def forward(self, input):

    # Preparation for group convolutions
    # Precompute a rotated stack of se2 kernels
    # With shape: [orientations_nb, kernelSizeH, kernelSizeW, orientations_nb,
    # channelsIN, channelsOUT]
    kernel_stack = rotate_gconv_kernels(self.kernel, self.periodicity, self.diskMask)
    
    #print("SE2N-SE2N ROTATED KERNEL SET SHAPE:", kernel_stack.shape)  # Debug

    # Group convolutions are done by integrating over [x,y,theta,input-channels] for each translation and rotation of the kernel
    # We compute this integral by doing standard 2D convolutions (translation part) for each rotated version of the kernel (rotation part)
    # In order to efficiently do this we use 2D convolutions where the theta
    # and input-channel axes are merged (thus treating the SE2 image as a 2D
    # feature map)

    # Prepare the input tensor (merge the orientation and channel axis) for
    # the 2D convolutions:
    input_tensor_as_if_2D = torch.reshape(input,
                                          [input.shape[0], self.orientations_nb * self.channelsIN, 
                                           int(input.shape[3]), int(input.shape[4])])

    #print("Input reshaped shape : ", input_tensor_as_if_2D.shape)

    # Reshape the kernels for 2D convolutions (orientation+channelsIN axis are
    # merged, rotation+channelsOUT axis are merged)
    kernels_as_if_2D = kernel_stack.permute(1, 2, 3, 4, 0, 5)
    kernels_as_if_2D = torch.reshape(kernels_as_if_2D, [self.kSize, self.kSize, 
                                                        self.orientations_nb * self.channelsIN, 
                                                        self.orientations_nb * self.channelsOUT])

    # Permute kernels : [nbOrientations * channelsOUT, nbOrientations * channelsIN, kernelSizeH, kernelSizeW]
    kernels_as_if_2D = kernels_as_if_2D.permute(3,2,0,1)

    # Perform the 2D convolutions
    layer_output = F.conv2d(input_tensor_as_if_2D.type(self.dtype), kernels_as_if_2D.type(self.dtype),padding=self.padding)

    # Reshape into an SE2 image (split the orientation and channelsOUT axis)
    layer_output = torch.reshape(layer_output, [layer_output.shape[0], self.orientations_nb, 
                                                self.channelsOUT, int(layer_output.shape[2]), int(layer_output.shape[3])])
    
    #print("OUTPUT SE2N ACTIVATIONS SHAPE:", layer_output.shape)  # Debug

    #return layer_output, kernel_stack
    return layer_output

class roto_block(nn.Module):

  def __init__(self, channelsIN, channelsOUT, kSize, orientations_nb,
               periodicity=2 * np.pi, diskMask=True, padding='same',
               dtype = torch.cuda.FloatTensor):
    super().__init__()

    self.orientations_nb = orientations_nb
    self.channelsOUT = channelsOUT

    self.lifting = lifting_block(channelsIN, channelsOUT, kSize, orientations_nb,
                                 periodicity=2 * np.pi, diskMask=True, padding=padding,
                                 dtype = torch.cuda.FloatTensor)
    
    self.gconv = gconv_block(channelsOUT, channelsOUT, kSize, orientations_nb,
                             periodicity=2 * np.pi, diskMask=True, padding=padding,
                             dtype = torch.cuda.FloatTensor)
    
    self.spatial_max_pool = spatial_max_pool(orientations_nb, channelsOUT, padding=0, stride=2)

    self.relu = nn.LeakyReLU(0.2, inplace=True)
    self.up = nn.Upsample(scale_factor=2, mode='nearest')
    self.bn = nn.BatchNorm2d(channelsOUT)

  def forward(self, input, skip):

    x = torch.cat([input, skip], axis=1)
    x = self.up(x)

    x = self.lifting(x)

    # Batch normalization, fusion des axes Orientations et ChannelsOUT
    x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4]))
    x = self.bn(x)
    x = self.relu(x)

    # Remise en forme tenseur SE2
    x = torch.reshape(x, (x.shape[0], self.orientations_nb, self.channelsOUT, x.shape[3], x.shape[4]))
    x = self.gconv(x)

    x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4]))
    x = self.bn(x)
    x = self.relu(x)

    # Remise en forme tenseur SE2
    x = torch.reshape(x, (x.shape[0], self.orientations_nb, self.channelsOUT, x.shape[3], x.shape[4]))

    # SHAPE [BatchSize, nbOrientations, ChannelsOUT, Height, Width]
    # Max sur la dimension des rotations
    x, id = torch.max(x,1)

    # SHAPE [BatchSize, ChannelsOUT, Height, Width]

    x = self.bn(x)
    x = self.relu(x)

    return x

class roto_block_noskip(nn.Module):

  def __init__(self, channelsIN, channelsOUT, kSize, orientations_nb,
               periodicity=2 * np.pi, diskMask=True, padding='same',
               dtype = torch.cuda.FloatTensor):
    super().__init__()

    self.orientations_nb = orientations_nb
    self.channelsOUT = channelsOUT

    self.lifting = lifting_block(channelsIN, channelsOUT, kSize, orientations_nb,
                                 periodicity=2 * np.pi, diskMask=True, padding=padding,
                                 dtype = torch.cuda.FloatTensor)
    
    self.gconv = gconv_block(channelsOUT, channelsOUT, kSize, orientations_nb,
                             periodicity=2 * np.pi, diskMask=True, padding=padding,
                             dtype = torch.cuda.FloatTensor)
    
    self.spatial_max_pool = spatial_max_pool(orientations_nb, channelsOUT, padding=0, stride=2)

    self.relu = nn.LeakyReLU(0.2, inplace=True)
    self.up = nn.Upsample(scale_factor=2, mode='nearest')
    self.bn = nn.BatchNorm2d(channelsOUT)

  def forward(self, input):

    x = self.up(input)

    x = self.lifting(x)

    # Batch normalization, fusion des axes Orientations et ChannelsOUT
    x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4]))
    x = self.bn(x)
    x = self.relu(x)

    # Remise en forme tenseur SE2
    x = torch.reshape(x, (x.shape[0], self.orientations_nb, self.channelsOUT, x.shape[3], x.shape[4]))
    x = self.gconv(x)

    x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4]))
    x = self.bn(x)
    x = self.relu(x)

    # Remise en forme tenseur SE2
    x = torch.reshape(x, (x.shape[0], self.orientations_nb, self.channelsOUT, x.shape[3], x.shape[4]))

    # SHAPE [BatchSize, nbOrientations, ChannelsOUT, Height, Width]
    # Max sur la dimension des rotations
    x, id = torch.max(x,1)

    # SHAPE [BatchSize, ChannelsOUT, Height, Width]

    x = self.bn(x)
    x = self.relu(x)

    return x

class build_hourglass_roto(nn.Module):
    
    def __init__(self,input_depth=32,output_depth=3,
                 num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
                 num_channels_skip=[4, 4, 4, 4, 4], 
                 filter_size_down=3, filter_size_up=3, filter_skip_size=1,
                 num_scales=5, up_samp_mode='bilinear', need1x1_up=True, need_sigmoid=True, pooling=False,
                 orientations_nb=4, filter_roto=5, etages_roto=[]):
        super().__init__()

        num_channels_down = [num_channels_down]*num_scales if isinstance(num_channels_down, int) else num_channels_down
        num_channels_up =   [num_channels_up]*num_scales if isinstance(num_channels_up, int) else num_channels_up
        num_channels_skip = [num_channels_skip]*num_scales if isinstance(num_channels_skip, int) else num_channels_skip

        self.num_channels_skip = num_channels_skip
        self.need_sigmoid = need_sigmoid

        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
        
        if pooling:
            stride = 1
        else:
            stride = 2

        self.num_scales = num_scales 
        
        attributes = []
        for i in range(num_scales):

          """ Encoder et Skip"""
          if i == 0:
            attributes.append(('e'+str(i+1),encoder_block(input_depth, num_channels_down[0],filter_size_down, stride).type(torch.cuda.FloatTensor)))
            #attributes.append(('e'+str(i+1),roto_block(input_depth, num_channels_down[0]//orientations_nb, filter_roto, orientations_nb).type(torch.cuda.FloatTensor)))
            if num_channels_skip[i] != 0: # Ne pas créer de bloc skip s'il n'en existe pas
              attributes.append(('s'+str(i+1),conv_block(num_channels_down[0], num_channels_skip[i], filter_skip_size).type(torch.cuda.FloatTensor)))
          else:
            attributes.append(('e'+str(i+1),encoder_block(num_channels_down[i-1], num_channels_down[i], filter_size_down, stride).type(torch.cuda.FloatTensor)))
            #attributes.append(('e'+str(i+1),roto_block(num_channels_down[i-1], num_channels_down[i]//orientations_nb, filter_roto, orientations_nb).type(torch.cuda.FloatTensor)))
            if num_channels_skip[i] != 0:
              attributes.append(('s'+str(i+1),conv_block(num_channels_down[i], num_channels_skip[i], filter_skip_size).type(torch.cuda.FloatTensor)))

          """ Decoder """
          if i == (num_scales-1):
            "Fond du réseau"
            if num_channels_skip[i] != 0:
              #attributes.append(('d'+str(i+1),roto_block(num_channels_down[i]+num_channels_skip[i], num_channels_up[i], filter_roto, orientations_nb).type(torch.cuda.FloatTensor)))
              attributes.append(('d'+str(i+1),decoder_block(num_channels_down[i]+num_channels_skip[i], num_channels_up[i], filter_size_up, up_sampling_mode=up_samp_mode, need1x1_up=need1x1_up).type(torch.cuda.FloatTensor)))
            else: # Pas de skip
              #attributes.append(('d'+str(i+1),roto_block_noskip(num_channels_down[i]+num_channels_skip[i], num_channels_up[i], filter_roto, orientations_nb).type(torch.cuda.FloatTensor)))
              attributes.append(('d'+str(i+1),decoder_noskip_block(num_channels_down[i]+num_channels_skip[i], num_channels_up[i], filter_size_up, up_sampling_mode=up_samp_mode, need1x1_up=need1x1_up).type(torch.cuda.FloatTensor)))
          #elif i == 3:
          #  if num_channels_skip[i] != 0:
          #    attributes.append(('d'+str(i+1),roto_block(num_channels_up[i+1]+num_channels_skip[i], num_channels_up[i], filter_roto, orientations_nb).type(torch.cuda.FloatTensor))) 
          #  else: # Pas de skip
          #    attributes.append(('d'+str(i+1),roto_block_noskip(num_channels_up[i+1]+num_channels_skip[i], num_channels_up[i], filter_roto, orientations_nb).type(torch.cuda.FloatTensor))) 
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
