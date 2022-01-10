import logging
import os
import time

import cv2
import GPUtil
import numpy as np
import rasterio
import torch
import buzzard as buzz

from src.semantic_segmentation.models import NETS
from src.semantic_segmentation.utils.image import (from_coord_to_patch, grouper,
                                               sliding_window)
from src.semantic_segmentation.utils.metrics import IoU, accuracy, f1_score

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def ponderated_map(size, overlap, lb, rb, tb, bb):
    a = np.ones((size, size))

    b = np.repeat(np.arange(1, overlap+1)[np.newaxis, np.newaxis], size, axis=1)[0] / overlap
    if not tb:
        a[:overlap] *= b.T
    if not lb:
        a[: ,:overlap] *= b
    if not rb:
        a[:, -overlap:] *= b[:, ::-1]
    if not bb:
        a[-overlap:] *= b.T[::-1]
    return a


class Inferer:
    def __init__(self, cfg, pretrain_file, img_file):
        self.cfg = cfg
        self.device = torch.device(self._set_device())
        self.net = torch.jit.load(pretrain_file, map_location=self.device).eval()
        c, *_, p = self.net.parameters()
        self.n_classes = p.shape[0] if len(p.shape) == 1 else p.shape[1]
        self.n_channels = c.shape[1]
        self.meta, self.proj = self.get_meta(img_file)

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

    def _infer_image(self, data):
        """infer for one image"""
        with torch.no_grad():
            img = data
            pred = np.zeros(
                img.shape[1:] + (self.n_classes,)
            )  # will contain the pred probabilities for each pixel
            occur_pix = np.zeros(
                (*img.shape[1:], 1)
            )  # will be used to average the probability when overlapping
            # Slides a window across the image
        for coords in grouper(
                self.cfg.BATCH_SIZE,
                sliding_window(img, step=self.cfg.STRIDE, window_size=self.cfg.WINDOW_SIZE),
            ):
            data_patches = [from_coord_to_patch(img, coords, self.device)]
            data_patches = torch.cat([*data_patches], dim=1).float()
            outs = self.net(data_patches)
            if isinstance(outs, tuple):
                outs = outs[0]
            outs = outs.data.cpu().numpy()

            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1, 2, 0))
                pred[x : x + w, y : y + h] += out
                occur_pix[x : x + w, y : y + h, :] += 1
        pred = pred / occur_pix
        return pred

    def get_meta(self, img):
        meta = rasterio.open(img).meta
        meta.update(dtype=np.uint8, compress="LZW", count=1)
        with buzz.Dataset().close as ds:
            proj = ds.aopen_raster(img).proj4_virtual
        return meta, proj

    def infer(self, img):
        self.net.eval()
        # Load data
        with rasterio.open(img) as src:
            data = torch.from_numpy(np.asarray(1 / 255 * src.read()[:3], dtype=np.float32))
            n_channels = data.shape[0]
            if self.n_classes + n_channels == self.n_channels:
                data = torch.cat([data, torch.zeros(self.n_classes, *data.shape[-2:])], dim=0)
        pred = self._infer_image(data)
        pred = np.argmax(pred, axis=-1).astype(np.uint8)

        pred[pred==2] = 3
        pred[pred==1] = 2
        pred[pred==0] = 1
        pred[pred==4] = 1
        pred[pred==5] = 0
        pred[pred==3] = 0
        return pred

    def compare(self, pred, gt):
        gt = rasterio.open(gt).read(1)
        
        # gt[gt==3] = 0
        # gt[gt==4] = 0
        # gt[gt==5] = 0
        # gt[gt==6] = 0
        # gt[gt==7] = 0

        gt[gt==2] = 3
        gt[gt==1] = 2
        gt[gt==0] = 1
        gt[gt==4] = 1
        gt[gt==5] = 0
        gt[gt==3] = 0

        acc = accuracy(pred, gt, ignore_indx=None)
        iou = IoU(pred, gt, self.n_classes, all_iou=True, ignore_indx=None)
        print("Accuracy", acc)
        print("IoU", iou)

    def save(self, image_file, output, outdir):
        outname = os.path.join(outdir, os.path.basename(image_file.replace('.', "_pred.")))
        with rasterio.open(outname, 'w', **self.meta) as out_file:
            out_file.write(output, indexes=1)
        self.polygonize(outname, outname.split('.')[0]+".geojson")

    def polygonize(self, input_file, output_file):
        """
        Polygonise a raster file
        Parameters
        ----------
        input_file: str Path to input raster
        output_file: str Path to output vector
        proj: str Projection
        """
        with buzz.Dataset(sr_work=self.proj, sr_fallback='WGS84').close as ds:
            ds.open_raster("raster", input_file)
            if os.path.isfile(output_file):
                os.remove(output_file)
            fields = [{"name": "class", "type": np.int32}]
            ds.create_vector("vector", output_file, 'polygon', driver="geojson", fields=fields)
            fp = ds["raster"].fp
            mask = ds["raster"].get_data()
            for class_idx in np.unique(mask):
                if class_idx != 0:
                    polygons = fp.find_polygons(mask == class_idx)
                    if not polygons:
                        continue
                    for poly in polygons:
                        ds["vector"].insert_data(poly, {"class": class_idx})