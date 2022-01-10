import logging
import time

import GPUtil
import numpy as np
import torch
import torch.jit as jit
import torch.optim as optim
from src.semantic_segmentation.models import NETS
from src.semantic_segmentation.utils.image import (from_coord_to_patch,
                                                   grouper, sliding_window)

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class Trainer:

    def __init__(self, cfg):
        self.number_gpus = 1
        self.cfg = cfg
        self.device = torch.device(self._set_device())
        print("Using Device: {}".format(self.device))

        params = {}
        self.net = self._get_net(cfg.NET_NAME)(
            in_channels=cfg.IN_CHANNELS,
            n_classes=cfg.N_CLASSES,
            pretrain=cfg.PRETRAIN,
            **params,
        )
        self.net = self.net.to(self.device)
        self.net_name = self.cfg.NET_NAME
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.OPTIM_BASELR * self.number_gpus,
            momentum=0.9,
            weight_decay=0.0005,
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, list(self.cfg.OPTIM_STEPS), gamma=0.1
        )

    def train(self, epochs, train_ids, test_ids, means, stds, pretrain_file=None):
        pass

    def test(self, test_ids, means, stds, sparsity=0, stride=None):
        pass

    def _get_net(self, net_name: str) -> torch.Tensor:
        return NETS[net_name]

    @staticmethod
    def _set_device():
        """Set gp device when cuda is activated. If code runs with mpi, """
        if not torch.cuda.is_available():
            return "cpu"
        device = "cuda:"
        if hvd is not None and hvd.local_rank() != 0:  # during mpi runs
            device += str(hvd.local_rank())
        else:
            for d, i in enumerate(GPUtil.getGPUs()):
                if i.memoryUsed < 4000:  # ie:  used gpu memory<900Mo
                    device += str(d)
                    break
                else:
                    if d + 1 == len(GPUtil.getGPUs()):
                        print("\033[91m \U0001F995  All GPUs are currently used by external programs. Using cpu. \U0001F996  \033[0m")
                        device = "cpu"
        return device


    def load_weights(self, path_weights: str) -> None:
        """Only to infer (doesn't load scheduler and optimizer state)."""
        print(path_weights)
        try:
            self.net = jit.load(path_weights, map_location=self.device)
        except:
            checkpoint = torch.load(path_weights)
            # self.net = jit.load(net_filename, map_location=self.device)
            self.net.load_state_dict(checkpoint["net"])
        
        logging.info("%s Weights loaded", time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime()))
        print("Weights succesfully loaded")

    def save_to_jit(self, name):
        self.net = self.net.to("cpu")
        n_channels = self.cfg.IN_CHANNELS
        dummy_tensor = torch.randn((self.cfg.BATCH_SIZE, n_channels, *self.cfg.WINDOW_SIZE))
        self.net.eval()
        torch_out = self.net(dummy_tensor)
        traced_script_module = torch.jit.trace(self.net, dummy_tensor)
        traced_script_module.save(name)
        self.net.to(self.device)
        return torch_out, dummy_tensor

    def _infer_image(self, stride, data, net, n_classes):
        """infer for one image"""
        with torch.no_grad():
            img = data
            pred = np.zeros(
                img.shape[1:] + (n_classes,)
            )  # will contain the pred probabilities for each pixel
            occur_pix = np.zeros(
                (*img.shape[1:], 1)
            )  # will be used to average the probability when overlapping
            # Slides a window across the image
        for coords in grouper(
                self.cfg.BATCH_SIZE,
                sliding_window(img, step=stride, window_size=self.cfg.WINDOW_SIZE),
            ):
            data_patches = [from_coord_to_patch(img, coords, self.device)]
            data_patches = torch.cat([*data_patches], dim=1).float()
            outs = net(data_patches)
            if isinstance(outs, tuple):
                outs = outs[0]
            outs = outs.data.cpu().numpy()

            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1, 2, 0))
                pred[x : x + w, y : y + h] += out
                occur_pix[x : x + w, y : y + h, :] += 1
        pred = pred / occur_pix
        return pred
