#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, tv_loss 
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch.nn.functional as F
from models.networks import CNN_decoder
from models.semantic_dataloader import VariableSizeDataset

from utils.dense_matching import DenseMatcher

import imageio
import numpy as np


from disk import DISK
from torch_dimcheck import dimchecked

import pickle

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


def load_image(img_path, crop_size = (None, None)):

    # Read the img
    img    = np.ascontiguousarray(imageio.imread(img_path))
    tensor = torch.from_numpy(img).to(torch.float32)

    if len(tensor.shape) == 2: # some images may be grayscale
            tensor = tensor.unsqueeze(-1).expand(-1, -1, 3)

    bitmap              = tensor.permute(2, 0, 1) / 255.
    name = os.path.basename(img_path)
    extensionless_fname = os.path.splitext(name)[0]

    image = Image(bitmap, extensionless_fname)

    #Resize the img
    if crop_size != (None, None):
        image = image.resize_to(crop_size)

    #add the bacth in the dim = 0
        # image [C,H,W] --> image [1,C,H,W] 
    # it corresponds the dimension demand of DISK 
    bitmap_o = torch.unsqueeze(image.bitmap, 0)

    return bitmap_o


def detect(query_path, ref_path):
    
    extract = model.descriptor
    img_ref = load_image(ref_path, (args.height, args.width)).to(DEV)
    img_query = load_image(query_path, (args.height, args.width) ).to(DEV)

    des_ref = extract(img_ref)
    des_query = extract(img_query)

    des_ref = torch.squeeze(des_ref, dim=0)
    des_query = torch.squeeze(des_query, dim=0)

    print("des ref size = ", des_query.shape)

    return des_ref, des_query

def match(des_ref, des_query):
    
    dense_matcher = DenseMatcher(des_ref, des_query)
    matches_ref, matches_query = dense_matcher.matching()

    match_p = {"ref": matches_ref, 
               "query": matches_query}
    
    if not os.path.exists('./match_points'):
        os.makedirs('./match_points')
    with open("./match_points/matchpoints.pkl", "wb") as fp:
        pickle.dump(match_p, fp)
             

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    #lp = ModelParams(parser)
    #op = OptimizationParams(parser)
    #pp = PipelineParams(parser)

    parser.add_argument("--img_query_path", type=str, default = None)
    parser.add_argument("--img_ref_path", type=str, default = None)
    parser.add_argument(
        '--height', default=None, type=int,
        help='rescaled height (px). If unspecified, image is not resized in height dimension'
    )
    parser.add_argument(
        '--width', default=None, type=int,
        help='rescaled width (px). If unspecified, image is not resized in width dimension'
    )
    default_model_path = os.path.split(os.path.abspath(__file__))[0] + '/depth-save.pth'
    parser.add_argument(
         '--model_pth', type=str, default=default_model_path,
        help="Path to the model's .pth save file"
    )


    args = parser.parse_args(sys.argv[1:])
    DEV   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CPU   = torch.device('cpu')

    state_dict = torch.load(args.model_pth, map_location='cpu')
    
    # compatibility with older model saves which used the 'extractor' name
    if 'extractor' in state_dict:
        weights = state_dict['extractor']
    elif 'disk' in state_dict:
        weights = state_dict['disk']
    else:
        raise KeyError('Incompatible weight file!')
    
    model = DISK(window=8, desc_dim=128)
    model.load_state_dict(weights)
    model = model.to(DEV)
    
    des_ref, des_query =  detect(args.img_query_path, args.img_ref_path)
    match(des_ref, des_query)


