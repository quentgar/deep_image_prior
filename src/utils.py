from PIL import Image

def format_image(img_path, mask_path, dim_div_by):
  img_pil, img_np = get_image(img_path, imsize)
  img_pil = crop_image(img_pil, dim_div_by)
  img_np = pil_to_np(img_pil)

  img_mask_pil, img_mask_np = get_image(mask_path, img_np.shape[1:])
  img_mask_pil = crop_image(img_mask_pil, dim_div_by)
  img_mask_np = 1 - pil_to_np(img_mask_pil)

  return img_np, img_mask_np

def crop_perso(img,d=32):
   new_size = (img.shape[1] - img.shape[1] % d, 
                img.shape[2] - img.shape[2] % d)

   img_cropped = img[:,
                     int((img.shape[1] - new_size[0])/2):int((img.shape[1] + new_size[0])/2),
                     int((img.shape[2] - new_size[1])/2):int((img.shape[2] + new_size[1])/2)]
                     
   return img_cropped
