from PIL import Image
import torch

def cross_entropy(t, p):
    return - torch.mean(t * torch.log2(p + 1e-10) + (1 - t) * torch.log2(1 - p + 1e-10))

def remove_transparency(im, bg_colour=(255, 255, 255)):
    '''reomve alpha channel or convert black and white image to RGB'''
    
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert('RGBA').split()[-1]
        bg = Image.new("RGB", im.size, bg_colour)
        bg.paste(im, mask=alpha)
        return bg
    elif im.mode == 'P':
        bg = Image.new("RGB", im.size, bg_colour)
        bg.paste(im)
        return bg
    elif im.mode == 'L':
        bg = Image.new("RGB", im.size, bg_colour)
        bg.paste(im)
        return bg
    else:
        return im