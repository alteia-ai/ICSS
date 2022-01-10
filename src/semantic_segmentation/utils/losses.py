"""
FESTA loss: Adapted from tensorflow to pytorch 
https://github.com/Hua-YS/Semantic-Segmentation-with-Sparse-Labels/blob/main/loss.py
"""
from itertools import filterfalse

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ Cross entropy loss """
    return nn.CrossEntropyLoss(weight)(input, target.long())


# The following code is from https://github.com/bermanmaxim/LovaszSoftmax

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in https://arxiv.org/pdf/1705.08790.pdf
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, only_present=False, per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(
            lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), only_present=only_present)
            for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()  # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    ll = iter(l)
    if ignore_nan:
        ll = filterfalse(np.isnan, ll)
    try:
        n = 1
        acc = next(ll)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(ll, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


#******************************FESTA LOSS******************************

# import keras.backend as K
# import tensorflow as torch #tensorflow.compat.v1 as torch
import torch

def torch_shuffle(t):
    idx = torch.randperm(t.nelement())
    t = t.view(-1)[idx].view(t.size())
    return t

def L_festa(y_pred):

    alpha = 0.5 # weight of neighbour in the feature space
    beta = 1.5 # weight of neighbour in the image space
    gamma = 1 # weight of far-away in the feature space

    sample_ratio = 0.01 # measure only sample_ratio % samples for computational efficiency
    y_pred = y_pred.permute(0,2,3,1)
    _, h, w, c = y_pred.shape
    batch_size = y_pred.shape[0]
    n_points = int(h*w*sample_ratio)
    y_pred_all_reshape = y_pred.reshape((batch_size, -1, c))
    random_idxs = torch_shuffle(torch.arange((h-2)*(w-2)))[:n_points]
    random_idxs = random_idxs + 513
    y_pred_reshape = y_pred_all_reshape[:,random_idxs]

    # ***************************** cosine similarity ***************************
    # calculating distance in the feature space
    xixj = torch.matmul(y_pred_reshape, y_pred_all_reshape.permute([0,2,1]))  # shape: BS * n_random_idxs * n_all_idxs
    similarity = xixj/(torch.unsqueeze(torch.norm(y_pred_reshape, dim=-1), dim = -1)*torch.unsqueeze(torch.norm(y_pred_all_reshape, dim=-1), dim = 1)+1e-8) + 1  # same shape. +1: Pour que ce soit entre 0 et 2
    faraway_feature = torch.min(similarity, dim = -1)[0] # feature with minimum similarity in the feaure space

    # ***************************** euclidean distance *************************** (1)
    distance = torch.unsqueeze(torch.square(torch.norm(y_pred_reshape, dim=-1)), dim=-1) - 2*xixj + torch.unsqueeze(torch.square(torch.norm(y_pred_all_reshape, dim=-1)), dim = 1) # shape: BS * n_random_idxs * n_all_idxs
    ind_diag = (torch.stack([torch.arange(n_points), random_idxs])).to(torch.int64) #shape: 2 * n_random_idxs
    distance[:,ind_diag[0], ind_diag[1]] = float('inf')
    neighbour_feature = torch.min(distance, dim = -1)[0] # feature with minimum distance in the feature space

     # ***************************** euclidean distance *************************** (2)
    
    # get indexes of 8-neighbouring pixels of the center pixel
    random_idxs_neighbors = [random_idxs - 1, random_idxs + 1, random_idxs - h -1, random_idxs - h, random_idxs - h + 1, random_idxs + h -1, random_idxs + h, random_idxs + h + 1]
    ind = torch.stack([torch.stack([torch.arange(n_points), i]).to(torch.int64) for i in random_idxs_neighbors], dim=0)
    dist_neighbor = torch.stack([distance[:,i[0], i[1]] for i in ind])

    neighbour_spatial = torch.min(dist_neighbor, dim = 0)[0] # feature with minimum distance in the image space

    delta = alpha*neighbour_feature+beta*neighbour_spatial+gamma*faraway_feature

    loss_reg = torch.mean(delta)
    return loss_reg