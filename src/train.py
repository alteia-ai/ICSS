"""
Train a network.
"""
import logging
import os
from os.path import join

import click
import numpy as np
import torch
from numpy import random

import git
from src.semantic_segmentation.classic_trainer import ClassicTrainer
from src.semantic_segmentation.config import config_factory

cwd = os.getcwd()
print(cwd)


@click.command()
@click.option("-d", "--dataset", help="Dataset on which to train/test.")
@click.option(
    "-c", "--config", default=None, help="Path to yaml configuration file.", required=False
)

def main(dataset, config):
    """Train a semantic segmentation network on GIS datasets."""
    # Set seeds for reproductibility
    cfg = config_factory(config)
    random.seed(42+cfg.ADD_SEED)
    torch.manual_seed(7+cfg.ADD_SEED)
    torch.backends.cudnn.deterministic = True

    output = cfg.SAVE_FOLDER
    model = cfg.PATH_MODELS

    if not os.path.exists(output):
        os.makedirs(output)

    if not os.path.exists(model):
        os.makedirs(model)

    print(dataset)
    dataset_name = os.path.basename(dataset.strip('/'))
    logging.basicConfig(
        format="%(message)s",
        filename="{}_Training_{}{}.log".format(join(cfg.SAVE_FOLDER, cfg.NET_NAME), dataset_name, cfg.ext),
        filemode="w",
        level=logging.INFO,
    )
    logging.info("Git commit: %s", git.Repo().head.object.hexsha)
    logging.info("Config : %s ", cfg)
    logging.info("Dataset, %s", dataset_name)

    net = ClassicTrainer(cfg, dataset=dataset)

    # Train network
    net.train(cfg.EPOCHS)

if __name__ == "__main__":
    main()
