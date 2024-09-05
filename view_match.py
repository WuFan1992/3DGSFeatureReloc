import argparse, os, imageio, torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import pickle


import numpy as np
import cv2

from utils.visual_common import MultiFigure
from torch_dimcheck import dimchecked

class Image:
    def __init__(self, bitmap: ['C', 'H', 'W'], fname: str, orig_shape=None):
        self.bitmap     = bitmap
        self.fname      = fname
        if orig_shape is None:
            self.orig_shape = self.bitmap.shape[1:]
        else:
            self.orig_shape = orig_shape

    def resize_to(self, shape):
        return Image(
            self._pad(self._interpolate(self.bitmap, shape), shape),
            self.fname,
            orig_shape=self.bitmap.shape[1:],
        )

    @dimchecked
    def to_image_coord(self, xys: [2, 'N']) -> ([2, 'N'], ['N']):
        f, _size = self._compute_interpolation_size(self.bitmap.shape[1:])
        scaled = xys / f

        h, w = self.orig_shape
        x, y = scaled

        mask = (0 <= x) & (x < w) & (0 <= y) & (y < h)
        
        return scaled, mask

    def _compute_interpolation_size(self, shape):
        x_factor = self.orig_shape[0] / shape[0]
        y_factor = self.orig_shape[1] / shape[1]

        f = 1 / max(x_factor, y_factor)

        if x_factor > y_factor:
            new_size = (shape[0], int(f * self.orig_shape[1]))
        else:
            new_size = (int(f * self.orig_shape[0]), shape[1])

        return f, new_size

    @dimchecked
    def _interpolate(self, image: ['C', 'H', 'W'], shape) -> ['C', 'h', 'w']:
        _f, size = self._compute_interpolation_size(shape)
        return F.interpolate(
            image.unsqueeze(0),
            size=size,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
    
    @dimchecked
    def _pad(self, image: ['C', 'H', 'W'], shape) -> ['C', 'h', 'w']:
        x_pad = shape[0] - image.shape[1]
        y_pad = shape[1] - image.shape[2]

        if x_pad < 0 or y_pad < 0:
            raise ValueError("Attempting to pad by negative value")

        return F.pad(image, (0, y_pad, 0, x_pad))





parser = argparse.ArgumentParser(
    description='Script for viewing the keypoints.h5 and matches.h5 contents',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--pkl_path', type=str, help='Path to pkl file where save the matching points')
parser.add_argument("--img_query_path", type=str, default = None)
parser.add_argument("--img_ref_path", type=str, default = None)
#parser.add_argument('image_path', help='Path to corresponding images')
#parser.add_argument(
#    '--image-extension', default='jpg', type=str,
#    help='Extension of the images'
#)
parser.add_argument(
    '--save', default=None, type=str,
    help=('If give a path, saves the visualizations rather than displaying '
          'them interactively')
)
#parser.add_argument(
#    'mode', choices=['keypoints', 'matches'],
#    help=('Whether to dispay the keypoints (in a single image) or matches '
#          '(across pairs)')
#)

args = parser.parse_args()

save_i = 1
def show_or_save():
    global save_i

    if args.save is None:
        plt.show()
        return
    else:
        path = os.path.join(os.path.expanduser(args.save), f'{save_i}.png')
        plt.savefig(path)
        print(f'Saved to {path}')
        save_i += 1
        plt.close()
"""
def view_keypoints(h5_path, image_path):
    keypoint_f = h5py.File(os.path.join(h5_path, 'keypoints.h5'), 'r')

    fname_to_id = {}
    for filename in tqdm(list(keypoint_f.keys())):
        keypoints = keypoint_f[filename][()]

        fname_with_ext = filename + '.' + args.image_extension
        path = os.path.join(image_path, fname_with_ext)
        if not os.path.isfile(path):
            raise IOError(f'Invalid image path {path}')

        image = imageio.imread(path)
        scale = 10 / max(image.shape)
        fig, ax = plt.subplots(figsize=(scale * image.shape[1], scale * image.shape[0]), constrained_layout=True)
        ax.axis('off')
        ax.imshow(image)
        ax.scatter(keypoints[:, 0], keypoints[:, 1], s=7, marker='o', color='white', edgecolors='black', linewidths=0.5)

        show_or_save()
"""

def view_matches(pkl_path, img_query_path, img_ref_path):

    with open(pkl_path, 'rb') as f:
        match_p = pickle.load(f)

    img_1 = imageio.imread(img_query_path)
    img_2 = imageio.imread(img_ref_path)

    img_1 = cv2.resize(img_1, (64, 32))
    img_2 = cv2.resize(img_2, (64, 32))

    bm_1 = torch.from_numpy(img_1)
    bm_2 = torch.from_numpy(img_2)


    """
    
    tensor_1 = bm_1.to(torch.float32)
    tensor_2 = bm_2.to(torch.float32)


    bitmap_1              = tensor_1.permute(2, 0, 1) / 255.
    name_query = os.path.basename(img_query_path)
    extensionless_fname_query = os.path.splitext(name_query)[0]

    bitmap_2              = tensor_2.permute(2, 0, 1) / 255.
    name_ref = os.path.basename(img_ref_path)
    extensionless_fname_ref = os.path.splitext(name_ref)[0]

    image_query = Image(bitmap_1, extensionless_fname_query)
    image_ref = Image(bitmap_2, extensionless_fname_ref)

    print("image ref shape = ", bitmap_1.shape)

    image_query = image_query.resize_to((64, 32))
    image_ref = image_ref.resize_to((64, 32))


    bm_1 = image_query.bitmap.permute(1,2,0)
    bm_2 = image_ref.bitmap.permute(1,2,0)

    print("bm_1 shape = ", bm_1.shape)


    bigger_x = max(bm_1.shape[0], bm_2.shape[0])
    bigger_y = max(bm_1.shape[1], bm_2.shape[1])

    padded_1 = F.pad(bm_1, (
                0, 0,
                0, bigger_y - bm_1.shape[1],
                0, bigger_x - bm_1.shape[0]
            ))
    padded_2 = F.pad(bm_2, (
                0, 0,
                0, bigger_y - bm_2.shape[1],
                0, bigger_x - bm_2.shape[0]
            ))

    
    ##########

    """
    

    bigger_x = max(bm_1.shape[0], bm_2.shape[0])
    bigger_y = max(bm_1.shape[1], bm_2.shape[1])

    padded_1 = F.pad(bm_1, (
                0, 0,
                0, bigger_y - bm_1.shape[1],
                0, bigger_x - bm_1.shape[0]
            ))
    padded_2 = F.pad(bm_2, (
                0, 0,
                0, bigger_y - bm_2.shape[1],
                0, bigger_x - bm_2.shape[0]
            ))
    
    fig = MultiFigure(padded_1, padded_2)

    ref_p = torch.tensor([list(e) for e in match_p["ref"] if e is not None])
    query_p = torch.tensor([list(e) for e in match_p["query"] if e is not None])

    fig.mark_xy(ref_p, query_p)

    show_or_save()

    ##########


"""
    bigger_x = max(bm_1.shape[0], bm_2.shape[0])
    bigger_y = max(bm_1.shape[1], bm_2.shape[1])

    padded_1 = F.pad(bm_1, (
                0, 0,
                0, bigger_y - bm_1.shape[1],
                0, bigger_x - bm_1.shape[0]
            ))
    padded_2 = F.pad(bm_2, (
                0, 0,
                0, bigger_y - bm_2.shape[1],
                0, bigger_x - bm_2.shape[0]
            ))

"""  
    

#if args.mode == 'keypoints':
#    pass
#    #view_keypoints(args.h5_path, args.image_path)
#elif args.mode == 'matches':
view_matches(args.pkl_path, args.img_query_path, args.img_ref_path)
