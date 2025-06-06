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
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from plyfile import PlyData, PlyElement
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, hyperparam=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    deform_vertices = []

    num_down_emb_c = hyperparam.min_embeddings
    num_down_emb_f = hyperparam.min_embeddings    
    
    count = 0
    total_time = 0
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if type(view.original_image) == type(None):
            if name == 'video':
                view.set_image()
            else:
                view.load_image()
        time1 = time()
        rendering = render(view, gaussians, pipeline, background, iter=iteration, num_down_emb_c=num_down_emb_c, num_down_emb_f=num_down_emb_f)["render"]
        time2 = time()
        total_time += (time2 - time1)
        render_images.append(to8b(rendering).transpose(1,2,0))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
        # render_list.append(rendering)
    
        if name in ["train", "test"]:
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
            # gt_list.append(gt)
        count +=1
        
    print("FPS:",(len(views)-1)/total_time)
    
    # count = 0
    # print("writing training images.")
    # if len(gt_list) != 0:
    #     for image in tqdm(gt_list):
    #         torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
    #         count+=1
    # count = 0
    # print("writing rendering images.")
    # if len(render_list) != 0:
    #     for image in tqdm(render_list):
    #         torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
    #         count +=1
    
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30, quality=8)


def render_sets(dataset : ModelParams, hyperparam, opt, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, duration=hyperparam.total_num_frames, loader=dataset.loader, opt=opt)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, hyperparam=hyperparam)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, hyperparam=hyperparam)
        if not skip_video:
            render_set(dataset.model_path, "video", scene.loaded_iter, scene.getVideoCameras(), gaussians, pipeline, background, hyperparam=hyperparam)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    
    # import sys
    # args = parser.parse_args(sys.argv[1:])
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), opt.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)
    # CUDA_VISIBLE_DEVICES=2 python render.py --model_path output/dynerf/coffee_martini_wo_cam13 --skip_train --configs arguments/dynerf/coffee_martini_wo_cam13.py