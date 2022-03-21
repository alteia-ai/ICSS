import logging
import os
import warnings
from abc import ABC, abstractmethod
from glob import glob
from typing import Any, Dict, List, Union

import numpy as np
import numpy.random as random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class SegDataLoader(Dataset, ABC):
    """ Abstract class serving as a Dataset for 
    pytorch training of interactive models on Gis
    datasets.
    """
    def __init__(self, dataset: str, cfg: Dict[str, Any]):
        """
        Parameters
        ----------
        dataset
            path to dataset
        cfg
            configuration dictionary
        """
        super().__init__()
        self.dataset = dataset
        self.cfg = cfg

    def __len__(self):
        """Defines the length of an epoch"""
        if self.train:
            return self.cfg.EPOCH_SIZE
        return len(self.test_ids)

    @abstractmethod
    def _load_data(self, i):
        return

    @abstractmethod
    def _data_augmentation(self, data):
        pass

    def get_loader(self, batch_size: int, workers: int = 0) -> torch.utils.data.DataLoader:
        """
        Parameters
        ----------
        train_set
            torch dataset
        batch_size
            Batch size
        workers
            Number of sub processes used in the process.

        Returns
        -------
        torch dataloader
        """
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, num_workers=workers, worker_init_fn=self.init_fn
        )

    def split_dataset(self, test_size: float):
        dataset_files = glob(os.path.join(self.dataset, "gts/*"))
        dataset_ids = np.arange(len(dataset_files))
        if test_size < 1:
            train_ids, test_ids = train_test_split(dataset_ids, test_size=test_size, random_state=42)
            train_ids = train_ids[:int(self.cfg.SUB_TRAIN * len(train_ids))]
        else:
            train_ids, test_ids = dataset_ids, dataset_ids
        return train_ids, test_ids

    @staticmethod
    def init_fn(worker_id):
        """ Initialize numpy seed for torch Dataloader workers."""
        random.seed(np.uint32(torch.initial_seed() + worker_id))
