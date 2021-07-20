from PIL import Image
import numpy as np

def format_image(img_path, dim_div_by):
  """ Load image as numpy array and transpose to C x W x H [0..1] """
  img_pil = Image.open(img_path1)
  ar = np.array(img_pil)

  if len(ar.shape) == 3:
    ar = ar.transpose(2,0,1)

  img_np = ar.astype(np.float32) / 255.

  img_np = crop_perso(img_np, dim_div_by)

  return img_np

def crop_perso(img,d=32):
   """ Make image dim divisible by d """
   new_size = (img.shape[1] - img.shape[1] % d, 
                img.shape[2] - img.shape[2] % d)

   img_cropped = img[:,
                     int((img.shape[1] - new_size[0])/2):int((img.shape[1] + new_size[0])/2),
                     int((img.shape[2] - new_size[1])/2):int((img.shape[2] + new_size[1])/2)]

   return img_cropped

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
