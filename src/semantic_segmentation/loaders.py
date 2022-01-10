import logging
import os
import warnings
from glob import glob
from typing import Any, Dict, List, Union

import cv2 as cv
import numpy as np
import numpy.random as random
import rasterio
from albumentations import (Compose, HorizontalFlip, HueSaturationValue,
                            RandomBrightnessContrast, RGBShift,
                            ShiftScaleRotate, VerticalFlip)
from icecream import ic
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.semantic_segmentation.SegDataLoader import SegDataLoader

rasterio_loader = logging.getLogger("rasterio")
rasterio_loader.setLevel(logging.ERROR)  # rasterio outputs warnings with some tiff files.


class RGBIncrementalDataset(SegDataLoader):
    def __init__(self, dataset, cfg, train=True, finetune=False, finetune_size=0.05):
        self.dataset = dataset
        self.cfg = cfg
        self.train = train
        self.finetune = finetune
        self.augmentation = cfg.TRANSFORMATION
        self.col_jit = cfg.COL_JIT  # Color jittering
        self.heavy_aug = int(cfg.HEAVY_AUG)  # Rotation, etc...
        assert cfg.IN_CHANNELS == 3
        self.train_ids, self.test_ids = self.split_dataset(cfg.test_size)
        gts = sorted(glob(os.path.join(dataset, "gts/*")))
        ext_gt = '.' + gts[0].split('.')[1]
        ext_imgs = '.' + sorted(glob(os.path.join(dataset, "imgs/*")))[0].split('.')[1]
        imgs = [file.replace("gts", "imgs").replace(ext_gt, ext_imgs) for file in gts]
        if train:
            self.gts = [gts[i] for i in self.train_ids]
            self.imgs = [imgs[i] for i in self.train_ids]
        else:
            self.gts = [gts[i] for i in self.test_ids]
            self.imgs = [imgs[i] for i in self.test_ids]
        self.means, self.stds = [0, 0, 0], [1, 1, 1]

    def _data_augmentation(self, features, labels, contours):
        """data augmentation"""
        transform = Compose(
            [HorizontalFlip(), VerticalFlip(), Compose([ShiftScaleRotate()], p=self.heavy_aug)], p=1
        )
        transform = transform(image=features, mask=labels, contours= contours)
        features = transform["image"]
        labels = transform["mask"]
        contours = transform["contours"] if contours is not None else None

        if self.col_jit:
            transform = Compose(
                [HueSaturationValue(p=0.3), RGBShift(p=0.2), RandomBrightnessContrast(p=0.5)], p=1
            )
            features = transform(image=features)["image"]
        return features, labels


    def _load_data(self, i):
        if self.train:
            #  Pick a random image and randomly set the coordinates of the crop
            random_id = random.randint(len(self.gts))
            with rasterio.open(self.gts[random_id]) as src:
                x_crop, y_crop = (
                    random.randint(max(1, src.shape[1] - self.cfg.WINDOW_SIZE[1])),
                    random.randint(max(1, src.shape[0] - self.cfg.WINDOW_SIZE[0])),
                )
                window = Window(x_crop, y_crop, self.cfg.WINDOW_SIZE[1], self.cfg.WINDOW_SIZE[0])
                del (src, x_crop, y_crop)
        else:
            # Not random for test and load the full images
            random_id = i
            window = None

        with rasterio.open(self.imgs[random_id]) as src:
            img = np.asarray(1 / 255 * src.read(window=window), dtype=np.float32)[:3].transpose(
                (1, 2, 0)
            )
            features = ((img - self.means) / self.stds).astype(np.float32)
        with rasterio.open(self.gts[random_id]) as src:
            labels = src.read(1, window=window)

            
        ### roads buildings
        # labels[labels==2] = 3
        # labels[labels==1] = 2
        # labels[labels==0] = 1
        # labels[labels==4] = 0
        # labels[labels==5] = 0
        # labels[labels==3] = 0
        if not self.finetune:
            labels[labels==2] = 0
        ### roads vegets (high / low)
        # labels[labels==1] = 4
        # labels[labels==0] = 1
        # labels[labels==4] = 0
        # labels[labels==5] = 0

        ### roads buildings & cars
        # labels[labels==2] = 5
        # labels[labels==3] = 5
        # labels[labels==4] = 3
        # labels[labels==1] = 2
        # labels[labels==0] = 1
        # labels[labels==5] = 0



        ### roads buildings SEMCITY
        # labels[labels==3] = 0
        # labels[labels==4] = 0
        # labels[labels==5] = 0
        # labels[labels==6] = 0
        # labels[labels==7] = 0
        # if not self.finetune:
        #     labels[labels==2] = 0
        return features, labels

    def __getitem__(self, i):
        """
        Sparsity and augmentation are applied if it was enabled in cfg.
        Returns
        -------
        Data and ground truth in the right tensor shape.

        """
        # load data
        with warnings.catch_warnings():
            # Remove warnings when image is not georeferenced.
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            features, labels = self._load_data(i)

        #  Data augmentation
        if self.train and self.augmentation:
            features, labels = self._data_augmentation(features, labels, None)

        features = features.transpose((2, 0, 1))
        return features, labels


class GTDataset(SegDataLoader):
    """Only load ground truth. Used to compute classes frequency."""
    def __init__(
            self,
            dataset: str,
            cfg: Dict[str, Any],
            ids: List[int]
    ):
        self.cfg = cfg
        self.dataset = dataset
        gts = sorted(glob(os.path.join(dataset, "gts/*")))
        self.gts = []
        self.gts = [gts[i] for i in ids]

    def _load_data(self, i):
        """Load data"""
        with rasterio.open(self.gts[i]) as src:
            labels = src.read(1)
        return labels

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, i):
        """
        Sparsity and augmentation are applied if it was enabled in cfg.
        Returns
        -------
        Data and ground truth in the right tensor shape.

        """
        # load data
        with warnings.catch_warnings():
            # Remove warnings when image is not georeferenced.
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            labels = self._load_data(i)
        return labels

    def split_dataset(self, test_size):
        # TODO: Refactoring neaded over here
        dataset_files = glob(os.path.join(self.dataset, "gts/*"))
        dataset_ids = np.arange(len(dataset_files))
        if test_size < 1:
            self.train_ids, self.test_ids = train_test_split(dataset_ids, test_size=test_size, random_state=42)
        else:
            self.train_ids = self.test_ids
        if len(self.train_ids) and len(self.test_ids):
            return self.train_ids, self.test_ids
        dataset_path = os.path.abspath(self.dataset)
        message = "Can't load dataset, propbably path is empty. \n {}".format(dataset_path)
        raise Exception(message)

    def compute_frequency(self):
        print("Computing weights...")
        weights = [[] for i in range(self.cfg.N_CLASSES)]
        labels = self.get_loader(1, 12)
        for gt in labels:
            for i in range(self.cfg.N_CLASSES):
                weights[i].append(np.where(gt == i)[0].shape[0])
        sum_pxls = np.sum(weights)
        weights = [1 / (np.sum(i) / sum_pxls) for i in weights]
        if self.cfg.N_CLASSES == 6:
            weights[-1] = min(weights)  # because clutter class is an ill-posed problem
        weights = np.asarray(weights)
        logging.info(f"Following weights have been computed: {weights}")
        ic(weights)
        return weights

    def _data_augmentation(self):
        pass
