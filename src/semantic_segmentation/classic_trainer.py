import logging
import os
import time
from glob import glob

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from src.semantic_segmentation.loaders import GTDataset, RGBIncrementalDataset
from src.semantic_segmentation.trainer import Trainer
from src.semantic_segmentation.utils.losses import CrossEntropy2d
from src.semantic_segmentation.utils.metrics import IoU, accuracy, f1_score
from tqdm import tqdm


class ClassicTrainer(Trainer):
    def __init__(self, cfg, train=True, dataset=None):
        super(ClassicTrainer, self).__init__(cfg)
        if train:
            self.train_dataset = RGBIncrementalDataset(dataset, self.cfg, finetune=False)
            self.gt_dataset = GTDataset(dataset, self.cfg, self.train_dataset.train_ids)
            logging.info(f"Train ids (len {len(self.train_dataset.imgs)}): {[os.path.basename(i) for i in self.train_dataset.imgs]}"
            )
        self.dataset = dataset
        test_dataset = RGBIncrementalDataset(dataset, self.cfg, train=False, finetune=False)
        logging.info(
            f"Test ids (len {len(test_dataset.imgs)}): {[os.path.basename(i) for i in test_dataset.imgs]}"
        )
        self.metrics = pd.DataFrame(data={i:[] for i in [os.path.basename(i) for i in test_dataset.imgs]}).T

    def train(self, epochs):
        """Train the network"""
        #  Initialization
        logging.info(
            "%s INFO: Begin training",
            time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime()),
        )

        iter_ = 0

        start_epoch, accu, iou, f1, train_loss, test_loss, losses = self._load_init()
        loss_weights = torch.ones(
            self.cfg.N_CLASSES, dtype=torch.float32, device=self.device
        )
        if self.cfg.WEIGHTED_LOSS:
            weights = self.gt_dataset.compute_frequency()
            loss_weights = (
                torch.from_numpy(weights).type(torch.FloatTensor).to(self.device)
            )

        train_loader = self.train_dataset.get_loader(
            self.cfg.BATCH_SIZE, self.cfg.WORKERS
        )
        for e in tqdm(range(start_epoch, epochs + 1), total=epochs):
            logging.info(
                "\n%s Epoch %s",
                time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime()),
                e,
            )
            self.scheduler.step()
            self.net.train()
            steps_pbar = tqdm(
                train_loader, total=self.cfg.EPOCH_SIZE // self.cfg.BATCH_SIZE
            )
            for data in steps_pbar:
                features, labels = data
                self.optimizer.zero_grad()
                features = features.float().to(self.device)
                labels = labels.float().to(self.device)
                output = self.net(features)
                if isinstance(output, tuple):
                    output, _, _ = output
                loss = CrossEntropy2d(output, labels, weight=loss_weights)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                iter_ += 1
                steps_pbar.set_postfix({"loss": loss.item()})
            train_loss.append(np.mean(losses[-1 * self.cfg.EPOCH_SIZE :]))
            logging.info(f"Train loss: {train_loss}")
            loss, iou_, acc_, f1_ = self.test()
            test_loss.append(loss)
            accu.append(acc_)
            iou.append(iou_ * 100)
            f1.append(f1_ * 100)
        # Save final state
        name =  "_".join([os.path.join(self.cfg.PATH_MODELS, self.net_name), os.path.basename(self.dataset), f"{self.cfg.ext}.pt"])
        self.save_to_jit(name)

    def test(self):
        logging.info(
            "%s INFO: Begin testing",
            time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime()),
        )
        csv_name = "{}_{}{}.csv".format(os.path.join(self.cfg.SAVE_FOLDER, self.cfg.NET_NAME), os.path.basename(self.dataset), self.cfg.ext)
        self.net.eval()
        loss, acc, iou, f1 = (
            [],
            [],
            [],
            [],
        )  # will contain the metric and loss calculated for each image
        test_dataset = RGBIncrementalDataset(self.dataset, self.cfg, train=False, finetune=False)
        test_images = test_dataset.get_loader(1, self.cfg.TEST_WORKERS)
        stride = self.cfg.STRIDE
        for iteration, (idx, data) in enumerate(tqdm(zip(test_dataset.test_ids, test_images), total=len(test_dataset.test_ids))):
            file_name = os.path.basename(sorted(glob(os.path.join(self.dataset, "gts", '*')))[idx])
            logging.info("Filename: %s", file_name)
            data = [i.squeeze(0) for i in data]
            img = data[:-1][0]
            gt = data[-1].cpu().numpy()
            pred_ = self._infer_image(stride, img, self.net, self.cfg.N_CLASSES)
            #  Computes the class with the highest probability
            pred = np.argmax(pred_, axis=-1)
            # Compute the metrics
            ignore_indx = None
            metric_acc = accuracy(pred, gt, ignore_indx=ignore_indx)
            metric_iou = IoU(pred, gt, self.cfg.N_CLASSES, all_iou=True, ignore_indx=ignore_indx)
            metric_f1 = f1_score(pred, gt, self.cfg.N_CLASSES, all=True, ignore_indx=ignore_indx)
            metric_iou, all_iou = metric_iou
            metric_f1, all_f1, weighted_f1 = metric_f1
            acc.append(metric_acc)
            iou.append(metric_iou)
            f1.append(metric_f1)

        logging.info("Mean IoU : " + str(np.nanmean(iou)))
        logging.info("Mean accu : " + str(np.nanmean(acc)))
        logging.info("Mean F1 : " + str(np.nanmean(f1)))
        return np.mean(loss), np.nanmean(iou), np.mean(acc), np.mean(f1)

    def _load_init(self):
        start_epoch = 1
        train_loss = []
        test_loss = []
        losses = []
        accu = []
        iou = []
        f1 = []
        return start_epoch, accu, iou, f1, train_loss, test_loss, losses
