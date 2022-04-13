import time
import warnings

import click
import numpy as np
import torch
from numpy import random

from src.semantic_segmentation.config import config_factory
from src.semantic_segmentation.inferer import Inferer


@click.command()
@click.option('-c', "--config", default=None, help="Path to  yaml configuration file.", required=True)
# @click.option("--gt", help="Path to the ground truth (if available)", default=None)
@click.option(
    "-p", "--pretrain_file", required=False, help="Path to a pretrained network", default=None
)
@click.option('-o', "--outdir", help="Folder where to save the output", default="data/inference/")
@click.option('-i', "--image", help="Path to source image", type=str, required=True)

def infer(config, pretrain_file, outdir, image):
    """Semantic segmentation demo."""
    cfg = config_factory(config)
    random.seed(42+cfg.ADD_SEED)
    torch.manual_seed(7+cfg.ADD_SEED)
    torch.backends.cudnn.deterministic = True
    net = Inferer(cfg, pretrain_file, image)
    tic = time.time()
    output = net.infer(image)
    toc = time.time()
    print(f"Inference time. {np.round(toc-tic, 1)} s.")
    gt = image.replace("imgs/", "gts/")
    net.compare(output, gt)
    net.save(image, output, outdir)
if __name__ == "__main__":
    infer()
