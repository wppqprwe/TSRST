import argparse

import numpy as np
from einops import reduce

from utils import load_image, save_image, load_mask
from image import crop_image
from PIL import Image
import tifffile
import cv2


def adjust_margins(img, pad, pad_value=None):
    extent = np.stack([[0, 0], img.shape[:2]]).T
    # make size divisible by pad without changing coords
    remainder = (extent[:, 1] - extent[:, 0]) % pad
    complement = (pad - remainder) % pad
    extent[:, 1] += complement
    if pad_value is None:
        mode = 'edge'
    else:
        mode = 'constant'
    img = crop_image(
            img, extent, mode=mode, constant_values=pad_value)
    return img


def reduce_mask(mask, factor):
    mask = reduce(
            mask.astype(np.float32),
            '(h0 h1) (w0 w1) -> h0 w0', 'mean',
            h1=factor, w1=factor) > 0.5
    return mask


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--mask', action='store_true')
    args = parser.parse_args()
    return args


def main():

    pad = 256
    args = get_args()

    if args.image:
        # load histology image
        img = load_image(args.prefix+'image-scaled.jpg')
        # img = cv2.imread('data/HG/Parent_Visium_Human_Glioblastoma_image.tif')
        # img = tifffile.imread("data/MK/Visium_HD_Mouse_Kidney_tissue_image.btf")
        # img = Image.open(args.prefix+'Visium_Human_Breast_Cancer_image.tif')
        # pad image with white to make dimension divisible by 256
        img = adjust_margins(np.array(img), pad=pad, pad_value=255)
        # save histology image
        save_image(img, f'{args.prefix}image.jpg')
        img = reduce(img.astype(float), '(h1 h) (w1 w) c -> h1 w1 c', 'mean',h=16, w=16).astype(np.uint8)
        save_image(img, f'{args.prefix}image-sc.jpg')

    if args.mask:
        # load tissue mask
        mask = load_mask(args.prefix+'mask-scaled.png')
        # pad mask with False to make dimension divisible by 256
        mask = adjust_margins(mask, pad=pad, pad_value=mask.min())
        # save tissue mask
        save_image(mask, f'{args.prefix}mask.png')
        # save_image(~mask, f'{args.prefix}mask-whitebg.png')
        mask = reduce_mask(mask, factor=16)
        save_image(mask, f'{args.prefix}mask-small.png')


if __name__ == '__main__':
    main()
