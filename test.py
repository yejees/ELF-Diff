import argparse
import os
import numpy as np
import torch as th
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from torchvision import utils
from pathlib import Path


def fft2(img):
    return th.fft.fftshift(th.fft.fft2(th.fft.ifftshift(img)))

def ifft2(img):
    return th.fft.ifftshift(th.fft.ifft2(th.fft.fftshift(img)))


def main():
    args = create_argparser().parse_args()
    args.save_dir = f'{args.save_dir}_{args.sample_method}'
    
    dist_util.setup_dist()
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    out_path = Path(args.save_dir) / 'output/predicted'
    out_path_undersampled = Path(args.save_dir) / 'output/undersampled'

    out_path.mkdir(parents=True, exist_ok=True)
    out_path_undersampled.mkdir(parents=True, exist_ok=True)

    print(out_path)
    
    file_name = args.data_path.split('/')[-1]
    
    data = np.load(args.data_path)
    data = th.from_numpy(data).to(dist_util.dev())
    
    mask = np.load(args.mask_path)
    mask = th.from_numpy(mask).to(dist_util.dev())
    
    idata = ifft2(data)
            
    samples = diffusion.p_sample_loop_single_elf(
        model,
        (1, 2, 512, 160),
        data,
        mask,
        args,
        clip_denoised=args.clip_denoised,
        # model_kwargs=model_kwargs,
        progress=True,
    )
    
    sample = th.mean(samples,dim=0,keepdim=True)

    utils.save_image(abs(sample[:,0:1]+1j*sample[:,1:2]), str(out_path / f'{file_name}.png'), nrow = 1,normalize=True)
    utils.save_image(abs(idata), str(out_path_undersampled / f'{file_name}.png'), nrow = 1)


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=1,
        range_t=0,
        use_ddim=False,
        base_samples="",
        model_path="",
        save_latents=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_method', type=str, default='MCG', help='One of [vanilla, MCG, repaint]')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory where the results will be saved')
    parser.add_argument('--ensemble_num', type=int, default=8, help='Number of ensemble samples')
    parser.add_argument('--mask_path', type=str, default='./data/mask/mask.npy', help='Path to the mask')
    parser.add_argument('--data_path', type=str, default='./data/undersampled_kspace.npy', help='Path to the undersampled kspace')
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()