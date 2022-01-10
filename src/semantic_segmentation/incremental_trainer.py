import logging
import os
import time
from copy import deepcopy
from glob import glob

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from torch import jit, nn, optim
from tqdm import tqdm

from src.semantic_segmentation.loaders import (GTDataset, RGBIncrementalDataset)
from src.semantic_segmentation.models import NETS
from src.semantic_segmentation.trainer import Trainer
from src.semantic_segmentation.utils.image import from_coord_to_patch, grouper, sliding_window
from src.semantic_segmentation.utils.losses import L_festa
from src.semantic_segmentation.utils.metrics import IoU, accuracy, f1_score


def freeze_bn(module):
    for name, m in module.named_children():
        if len(list(m.children())) > 0:
            freeze_bn(m)
        if "bn" in name: 
            m.weight.requires_grad = False 
            m.bias.requires_grad = False 

palette={
    0: (255, 255, 255),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (0, 255, 255)
}

class IncrementalTrainer(Trainer):
    def __init__(self, cfg, dataset, pretrain_file):
        super(IncrementalTrainer, self).__init__(cfg)
        if cfg.NEW_CLASSES != 1:
            raise NotImplementedError()
        self.dataset = dataset
        test_dataset = RGBIncrementalDataset(dataset, self.cfg, train=False, finetune=True)
        logging.info(
            f"Test ids (len {len(test_dataset.imgs)}): {[os.path.basename(i) for i in test_dataset.imgs]}"
        )
        self.metrics = pd.DataFrame(data={i:[] for i in [os.path.basename(i) for i in test_dataset.imgs]}).T
        cfg.N_CLASSES += cfg.NEW_CLASSES
        state_dict =jit.load(pretrain_file).eval().state_dict()
        if state_dict[list(state_dict.keys())[-1]].shape[0] == cfg.N_CLASSES:
            # ie we load a network already designed for fine tuning
            self.net =jit.load(pretrain_file, map_location=self.device).eval()
            self.memory_net = self.net
            # pass
        elif state_dict[list(state_dict.keys())[-1]].shape[0] == cfg.N_CLASSES - cfg.NEW_CLASSES:
            self.net.load_state_dict(state_dict)
            seed = torch.get_rng_state()
            self.memory_net = NETS[cfg.NET_NAME](cfg.IN_CHANNELS, cfg.N_CLASSES-cfg.NEW_CLASSES, cfg.PRETRAIN).eval().to(self.device)
            self.memory_net.load_state_dict(state_dict)
            torch.set_rng_state(seed)
            if cfg.FREEZE:
                for param in self.net.parameters():
                    param.requires_grad = False
                    # to only freeze the encoder
                    if cfg.ENCODER_ONLY:
                        if len(param.shape) == 4 and param.shape[0] > param.shape[1]:
                            break
            if cfg.NET_NAME == "LinkNet34":
                self.net.tp_conv2 = torch.nn.ConvTranspose2d(32, cfg.N_CLASSES, 2, 2, 0)
                self.net.to(self.device)
                self.state_dict = deepcopy(self.net.state_dict())
            else:
                raise NotImplementedError()
        else:
            raise Exception(f"Wrong number of classes. Got {state_dict[list(state_dict.keys())[-1]].shape[0]} classes. ")
        if cfg.SDR_opts.ORIGINAL_SPACE_opts.PROTOTYPE_CHANNELS != 512 and cfg.SDR_opts.ORIGINAL_SPACE_opts.PROTOTYPE_CHANNELS != 3:
            raise Exception()
        self.previous_prototypes_ = torch.zeros(self.cfg.N_CLASSES, cfg.SDR_opts.ORIGINAL_SPACE_opts.PROTOTYPE_CHANNELS).to(self.device)
        self.previous_prototypes = torch.zeros(self.cfg.N_CLASSES, cfg.SDR_opts.ORIGINAL_SPACE_opts.PROTOTYPE_CHANNELS).to(self.device)  

    def _compute_prototypes(self, inputs, labels):
        prototypes = []
        masked_inputs = []
        for i in range(self.cfg.N_CLASSES):
            mask = (torch.unsqueeze(labels, dim=1) == i)
            masked_inputs.append(inputs * mask)
            prototypes.append(torch.sum(masked_inputs[i], dim=(0, 2, 3)) /mask.sum(dim=(0, 2,3)))
        return prototypes, masked_inputs

    def _generate_annots(self, gt, initial_pred, sparse_gt=None):
        """ to gennerate sparse/pseudo annotations"""
        logging.info(f"\n Genrate new annotations.")
        weights = torch.ones(self.cfg.N_CLASSES).to(self.device)
        if sparse_gt is None:
            flat_gt = gt.reshape((-1))
            sparse_gt = np.full_like(flat_gt, 255, dtype=np.uint8)
            supp_class = 0 # ie does NOT generate pseudo labels also for bg and new class if PSEUDOLABELS
            mult_pseudolabels = 10
        else:
            flat_gt = gt.reshape((-1))
            sparse_gt = sparse_gt.reshape((-1))
            supp_class = 1 # ie generate pseudo labels also for bg and new class if PSEUDOLABELS
            mult_pseudolabels = 1
            logging.info(f"Currently {sum(sparse_gt!=255)} annotations.")
        if self.cfg.TRAIN_ON_SPARSE_GT:
            for i in range(self.cfg.N_CLASSES):
                if self.cfg.PSEUDOLABELS and i < self.cfg.N_CLASSES - self.cfg.NEW_CLASSES + supp_class and i > 0 - supp_class:
                    probs = (initial_pred[i] > 0.99).reshape((-1)).numpy()
                    probs[sparse_gt!=255] = 0
                    probs = probs / np.sum(probs)  # normalize
                    sparse_points = np.random.choice(np.prod(gt.shape), mult_pseudolabels*self.cfg.N_POINTS, replace=False, p=probs)
                else:
                    probs = flat_gt == i
                    probs[sparse_gt!=255] = 0
                    probs = probs / np.sum(probs)  # normalize
                    sparse_points = np.random.choice(np.prod(gt.shape), self.cfg.N_POINTS, replace=False, p=probs)
                sparse_gt[sparse_points] = i
                logging.info(f"Class {i}, annots well/bad classified: {sum(flat_gt[sparse_gt==i]==i)} / {sum(flat_gt[sparse_gt==i]!=i)}")
                if self.cfg.WEIGHTED_INCREMENTAL_LOSS:
                    weights[i] = np.sum(sparse_gt==i)
            weights = 1 / (weights / torch.sum(weights))
            logging.info(f"Cross entropy weights: {weights}")
            sparse_gt = sparse_gt.reshape(*gt.shape)
        else:
            sparse_gt = gt
        return sparse_gt, weights

    def _compute_loss(self, features, labels, initial_pred, optimizer, loss_weights, d_losses, ignore_index=None, iteration=1):
        output = self.net(features)
        output, encoded_features, decoded_features = output
        pred = torch.argmax(output.detach(), dim=1)
        acc = int(torch.sum(pred== labels).cpu().numpy()) / labels.shape.numel()
        ce_loss = nn.CrossEntropyLoss(weight=loss_weights, ignore_index=ignore_index)(output, labels)
        if self.cfg.SDR:
            if not self.cfg.SDR_opts.LATENT_SPACE:  
                feats_up = torch.nn.functional.interpolate(encoded_features, scale_factor=32) # size=self.cfg.WINDOW_SIZE)
                if self.cfg.SDR_opts.ORIGINAL_SPACE_opts.PROTOTYPE_CHANNELS == 512:
                    prototypes, masked_inputs = self._compute_prototypes(feats_up, labels)
                elif self.cfg.SDR_opts.ORIGINAL_SPACE_opts.PROTOTYPE_CHANNELS == 3:
                    prototypes, masked_inputs = self._compute_prototypes(decoded_features, labels)
            else:
                encoded_labels = torch.squeeze(torch.nn.functional.interpolate(torch.unsqueeze(labels.float(), dim=1), scale_factor=1/32, mode='nearest', recompute_scale_factor=False), dim=1)
                prototypes, masked_inputs = self._compute_prototypes(encoded_features, encoded_labels)

            proto_loss = lambda x,y: torch.mean(torch.mean((x-y)**2, dim=-1))  
            #  CrossEntropy2d(output, labels, weight=loss_weights)
            match_loss, attraction_loss, repulsive_loss = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)
            # Match loss
            for i in range(self.cfg.N_CLASSES):
                match_loss_ = torch.nn.MSELoss()(prototypes[i], self.previous_prototypes[i]) / self.cfg.N_CLASSES
                if match_loss_ == match_loss_:
                    match_loss += match_loss_
                self.previous_prototypes_[i] = prototypes[i].detach()
                # Attraction loss
                masked_inputs[i] = masked_inputs[i].flatten(-2).transpose(-1,0).transpose(1,2)
                # attraction_loss_ = torch.nn.MSELoss()(masked_inputs[i], prototypes[i]) / self.cfg.N_CLASSES
                attraction_loss_ = proto_loss(masked_inputs[i], prototypes[i]) / self.cfg.N_CLASSES
                if attraction_loss_ == attraction_loss_:
                    attraction_loss += attraction_loss_
                # Repulsive
                for j in range(self.cfg.N_CLASSES):
                    if j != i:
                        repulsive_loss_ = 1/proto_loss(masked_inputs[i], prototypes[j]) / (self.cfg.N_CLASSES * (self.cfg.N_CLASSES-1))
                        if repulsive_loss_ == repulsive_loss_:
                            repulsive_loss += repulsive_loss_
            loss = ce_loss + repulsive_loss + match_loss + attraction_loss
        else:
            loss = ce_loss
        if self.cfg.DISCA and initial_pred is not None:
            predicted_labels = torch.argmax(initial_pred, dim=1)
            predicted_labels[predicted_labels==0] = ignore_index
            reg = nn.CrossEntropyLoss(ignore_index=ignore_index)(output, predicted_labels)
            d_losses["DISCA"] += reg.item()
            loss = loss + reg / iteration
        if self.cfg.FESTA:
            reg_festa = L_festa(decoded_features)
            loss = loss + reg_festa
            d_losses["festa"] += reg_festa.item()
        if self.cfg.PodNet:
            old_output = self.memory_net(features)
            _, old_encoded_features, _ = old_output  # old_decoded_features
            pooled_encoded_features = torch.cat([torch.sum(encoded_features, dim=-2), torch.sum(encoded_features, dim=-1)], dim=-1)
            old_pooled_encoded_features = torch.cat([torch.sum(old_encoded_features, dim=-2), torch.sum(old_encoded_features, dim=-1)], dim=-1)
            reg_distil = nn.MSELoss()(pooled_encoded_features, old_pooled_encoded_features)
            loss = loss + reg_distil
            d_losses["PodNet"] += reg_distil.item()
        loss.backward()
        d_losses["acc"] += acc
        d_losses["ce"] += ce_loss.item()
        if self.cfg.SDR:
            d_losses["match"] += match_loss.item()
            d_losses["attract"] += attraction_loss.item()
            d_losses["repulse"] += repulsive_loss.item()
        optimizer.step()
        for i in range(self.cfg.N_CLASSES):
            with torch.no_grad():
                    self.previous_prototypes[i] = self.previous_prototypes_[i]
        return loss, d_losses

    def test_incremental(self):
        ############## same code than in self.test() ##############
        logging.info(
            "%s INFO: Begin testing",
            time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime()),
        )
        self.net.eval()
        freeze_bn(self.net)
        loss, acc, iou, f1 = (
            [],
            [],
            [],
            [],
        )
        test_dataset = RGBIncrementalDataset(self.dataset, self.cfg, train=False, finetune=True)
        test_images = test_dataset.get_loader(1, self.cfg.TEST_WORKERS)
        stride = self.cfg.STRIDE
        lr = self.cfg.CL_LR
        for iteration, (idx, data) in enumerate(tqdm(zip(test_dataset.test_ids, test_images), total=len(test_dataset.test_ids))):
            file_name = os.path.basename(sorted(glob(os.path.join(self.dataset, "gts", '*')))[idx])
            logging.info("\n Filename: %s", file_name)
            data = [i.squeeze(0) for i in data]
            img = data[:-1][0]
            gt = data[-1].cpu().numpy()  
            ############# new code ##############
            target_classes = self.cfg.N_CLASSES if self.memory_net is self.net else self.cfg.N_CLASSES-self.cfg.NEW_CLASSES
            initial_pred = self._infer_image(stride, img, self.memory_net, target_classes)
            initial_pred = torch.from_numpy(initial_pred.transpose((2,0,1))[np.newaxis])
            initial_pred = torch.nn.Softmax2d()(initial_pred)[0]

            init_pred = self._infer_image(stride, img, self.net, self.cfg.N_CLASSES)
            pred = np.argmax(init_pred, axis=-1)
            ignore_indx = None
            metric_iou = IoU(pred, gt, self.cfg.N_CLASSES, all_iou=True, ignore_indx=ignore_indx)
            metric_f1 = f1_score(pred, gt, self.cfg.N_CLASSES, all=True, ignore_indx=ignore_indx)
            metric_iou, all_iou = metric_iou
            metric_f1, all_f1, weighted_f1 = metric_f1
            metric_acc = accuracy(pred, gt, ignore_indx=ignore_indx)
            logging.info("Before retraining")
            logging.info("IoU: %s", metric_iou)
            for c, i in enumerate(all_iou):
                logging.info(f"Class {c}, IoU {i}")
            logging.info("F1: %s", metric_f1)
            logging.info("Accuracy: %s", metric_acc)
            if self.memory_net is self.net:
                continue # To only compute initial accuracy with new network

            sparse_gt, weights = self._generate_annots(gt, initial_pred)
            steps = self.cfg.CL_STEPS
            print("learn\n")
            optimizer = optim.Adam(self.net.parameters(), lr)
            IOU = metric_iou
            ii = 0
            ious = []
            while IOU < 1:
                sparse_target = torch.from_numpy(sparse_gt[np.newaxis]).to(self.device)
                d_losses = {
                        "acc": 0,
                        "DISCA": 0,
                        "ce": 0,
                        "match": 0,
                        "attract": 0,
                        "repulse": 0,
                        "festa": 0,
                        "PodNet": 0,
                    }
                L = []
                iteration_ = 0
                for step in range(steps):
                    bs = self.cfg.BATCH_SIZE
                    for coords in grouper(
                            bs,
                            sliding_window(img, step=self.cfg.STRIDE, window_size=self.cfg.WINDOW_SIZE),
                        ):
                        optimizer.zero_grad()
                        target = from_coord_to_patch(sparse_target, coords, self.device).long()
                        data_patches = from_coord_to_patch(img, coords, self.device).float()
                        initial_pred_patches = from_coord_to_patch(initial_pred, coords, self.device).float() if self.cfg.DISCA else None
                        target = target[:,0]
                        loss, d_losses = self._compute_loss(data_patches, target, initial_pred_patches, optimizer, weights, d_losses, ignore_index=255, iteration=ii+1)
                        iteration_ += 1
                        L.append(loss.item())

                pred_ = self._infer_image(stride, img, self.net, self.cfg.N_CLASSES)
                pred = np.argmax(pred_, axis=-1)
                for key, value in d_losses.items():
                    logging.info(f"Mean {key} loss: {value / iteration_}")
                ignore_indx = None
                metric_iou = IoU(pred, gt, self.cfg.N_CLASSES, all_iou=True, ignore_indx=ignore_indx)
                metric_f1 = f1_score(pred, gt, self.cfg.N_CLASSES, all=True, ignore_indx=ignore_indx)
                metric_iou, all_iou = metric_iou
                metric_f1, all_f1, weighted_f1 = metric_f1
                metric_acc = accuracy(pred, gt, ignore_indx=ignore_indx)
                acc.append(metric_acc)
                iou.append(metric_iou)
                f1.append(metric_f1)
                logging.info(f"After retraining phase {ii}")
                logging.info(f"Loss: {np.nanmean(L)}")
                logging.info("IoU: %s", metric_iou)
                for c, i in enumerate(all_iou):
                    logging.info(f"Class {c}, IoU {i}")
                logging.info("F1: %s", metric_f1)
                IOU = metric_iou
                ious.append(IOU)
                self.metrics.loc[file_name, f"{ii}_acc"] = metric_acc
                self.metrics.loc[file_name, f"{ii}_IoU"] = metric_iou
                self.metrics.loc[file_name, f"{ii}_F1"] = metric_f1
                self.metrics.loc[file_name, f"{ii}_F1_weighted"] = weighted_f1
                for c, i in enumerate(all_iou):
                    self.metrics.loc[file_name, f"{ii}_IoU_class_{c}"] = i
                for c, i in enumerate(all_f1):
                    self.metrics.loc[file_name, f"{ii}_F1_class_{c}"] = i
                    csv_name = "{}_{}{}.csv".format(os.path.join(self.cfg.SAVE_FOLDER, self.cfg.NET_NAME), os.path.basename(self.dataset), self.cfg.ext)
                self.metrics.to_csv(csv_name)
                ii += 1
                patience = self.cfg.PATIENCE
                if (len(ious) >= patience and ious[-self.cfg.PATIENCE] == max(ious)) or len(ious) == 100:
                    logging.info(f"Max attained {patience} iterations ago. Process new image.") 
                    self.net.load_state_dict(self.state_dict)
                    self.state_dict = deepcopy(self.net.state_dict())
                    break
