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
from torch.utils.data import DataLoader

from encoders.sam_encoder.segment_anything import sam_model_registry, SamPredictor
import cv2
from utils.dense_matching import DenseMatcher


def get_embedded_image(predictor: SamPredictor, img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load '{img}' as an image, Error...")
        return
    
    predictor.set_image(img)
    image_embedding_tensor = torch.tensor(predictor.get_image_embedding().cpu().numpy()[0])
    ###
    img_h, img_w, _ = img.shape
    _, fea_h, fea_w = image_embedding_tensor.shape
    cropped_h = int(fea_w / img_w * img_h + 0.5)
    image_embedding_tensor_cropped = image_embedding_tensor[:, :cropped_h, :]
    return image_embedding_tensor_cropped

def get_embedded_images(model_type, img_path_query, img_path_ref, model):
    # Load model 
    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=model)
    sam.to("cuda")
    predictor = SamPredictor(sam)

    img_query_embedding = get_embedded_image(predictor, img_path_query)
    img_ref_embedding = get_embedded_image(predictor, img_path_ref)

    return img_query_embedding, img_ref_embedding



def matching(dataset, opt, pipe, checkpoint, model_type, img_query_path, img_ref_path, model):
    first_iter = 0
    #gaussians = GaussianModel(dataset.sh_degree)
    #scene = Scene(dataset, gaussians)

    img_query_embedding, img_ref_embedding = get_embedded_images(model_type, img_query_path, img_ref_path, model)

    dense_matcher = DenseMatcher(img_query_embedding, img_ref_embedding)
    query_matching_points, projection_matching_points = dense_matcher()

    print("query matching point is ", query_matching_points)
    
    """
    
    # 2D semantic feature map CNN decoder
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    gt_feature_map = viewpoint_cam.semantic_feature.cuda()
    feature_out_dim = gt_feature_map.shape[0]
    print("feature_out_dim = ", feature_out_dim)

    
    # speed up for SAM
    if dataset.speedup:
        feature_in_dim = int(feature_out_dim/2)
        cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
        cnn_decoder_optimizer = torch.optim.Adam(cnn_decoder.parameters(), lr=0.0001)


    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1


    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        

        feature_map, image, viewspace_point_tensor, visibility_filter, radii = render_pkg["feature_map"], render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        gt_feature_map = viewpoint_cam.semantic_feature.cuda()
        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) 
  
        if dataset.speedup:
            feature_map = cnn_decoder(feature_map)
        Ll1_feature = l1_loss(feature_map, gt_feature_map) 
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 1.0 * Ll1_feature 

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

          
  

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if dataset.speedup:
                    cnn_decoder_optimizer.step()
                    cnn_decoder_optimizer.zero_grad(set_to_none = True)


        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None
            """



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--model_type",type=str,required=True,help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",)
    parser.add_argument("--img_query_path", type=str, default = None)
    parser.add_argument("--img_ref_path", type=str, default = None)
    parser.add_argument("--model",type=str,required=True, help="The path to the SAM checkpoint to use for mask generation.",)


    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    print ("source path = ", lp._source_path )
    matching(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, args.model_type, args.img_query_path, args.img_ref_path, args.model)

    # All done
    print("\nTraining complete.")
