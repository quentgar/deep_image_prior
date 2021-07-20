from PIL import Image
import numpy as np

def crop_perso(img,d=32):
   """ Make image dim divisible by d """
   new_size = (img.shape[1] - img.shape[1] % d, 
                img.shape[2] - img.shape[2] % d)

   img_cropped = img[:,
                     int((img.shape[1] - new_size[0])/2):int((img.shape[1] + new_size[0])/2),
                     int((img.shape[2] - new_size[1])/2):int((img.shape[2] + new_size[1])/2)]

   return img_cropped

def format_image(img_path, dim_div_by):
  """ Load image as numpy array and transpose to C x W x H [0..1] """
  img_pil = Image.open(img_path)
  ar = np.array(img_pil)

  if len(ar.shape) == 3:
    ar = ar.transpose(2,0,1)

  img_np = ar.astype(np.float32) / 255.

  img_np = crop_perso(img_np, dim_div_by)

  return img_np

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]

def get_params(net):
   """ Return net parameters """
   return [x for x in net.parameters()]

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos'):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    
    plt.show()
    
    return grid

def optimize(parameters, closure, LR, num_iter):
  """ Optimize net with Adam """
  print('Starting optimization with ADAM')
  optimizer = torch.optim.Adam(parameters, lr=LR)
  
  for j in range(num_iter):
    optimizer.zero_grad()
    closure()
    optimizer.step()

def optimize_joint(parameters1, parameters2, closure, LR_inp, LR_rec, num_iter, ind_iter=1):
  """ Parallel optimization of inpainting and registration """
  print('Starting optimization with ADAM')
  optimizer_inpainting = torch.optim.Adam(parameters1, lr=LR_inp)
  optimizer_recalage = torch.optim.Adam(parameters2, lr=LR_rec)

  iter = num_iter // ind_iter

  for j in range(iter):
      for i in range(ind_iter):
        # Optimiser paramètres inpainting
        optimizer_inpainting.zero_grad()
        closure()
        optimizer_inpainting.step()

      for i in range(ind_iter):
        # Optimiser paramètres recalage
        optimizer_recalage.zero_grad()
        closure()
        optimizer_recalage.step()
