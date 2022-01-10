import numpy as np
import cv2
from tqdm import tqdm
from time import time
from sklearn.metrics import f1_score as sk_f1


def accuracy(input, target, ignore_indx=None):
    """Computes the total accuracy"""
    if ignore_indx is None:
        return 100 * float(np.count_nonzero(input == target)) / target.size
    else:
        mask = input == target
        mask[np.where(target == ignore_indx)] = False
        total = np.sum(np.where(target != ignore_indx, 1, 0))
        return 100 * np.sum(mask) / total


def IoU(pred, gt, n_classes, all_iou=False, ignore_indx=None):
    """Computes the IoU by class and returns mean-IoU"""
    # print("IoU")
    iou = []
    for i in range(n_classes):
        if i == ignore_indx:
            continue
        if np.sum(gt == i) == 0:
            iou.append(np.NaN)
            continue
        TP = np.sum(np.logical_and(pred == i, gt == i))
        FP = np.sum(np.logical_and(pred == i, gt != i))
        FN = np.sum(np.logical_and(pred != i, gt == i))
        iou.append(TP / (TP + FP + FN))
    # nanmean: if a class is not present in the image, it's a NaN
    result = [np.nanmean(iou), iou] if all_iou else np.nanmean(iou)
    return result


def f1_score(pred, gt, n_classes, all=False, ignore_indx=None):
    f1 = []
    for i in range(n_classes):
        if i == ignore_indx:
            continue
        if np.sum(gt == i) == 0:
            f1.append(np.NaN)
            continue
        TP = np.sum(np.logical_and(pred == i, gt == i))
        FP = np.sum(np.logical_and(pred == i, gt != i))
        FN = np.sum(np.logical_and(pred != i, gt == i))
        prec = TP / (TP + FP)
        recall = TP / (TP + FN)
        result = 2 * (prec * recall) / (prec + recall)
        f1.append(result)
    result = [np.nanmean(f1), f1] if all else np.nanmean(f1)
    if all:
        flat_pred = pred.reshape(-1)
        flat_gt = gt.reshape(-1)
        if ignore_indx is not None:
            flat_pred = flat_pred[np.where(flat_gt != ignore_indx)]
            flat_gt = flat_gt[np.where(flat_gt != ignore_indx)]
        f1_weighted = sk_f1(flat_gt, flat_pred, average="weighted")
        result.append(f1_weighted)
    return result


def xy_from_contour(img, contour):
    """Returns the coordinates of the points inside the contour ine the image. Binary image required. May be optimized."""
    mask = np.zeros((img.shape[0], img.shape[1], 3))
    mask = cv2.drawContours(mask, [contour], 0, (1, 0, 0), thickness=cv2.FILLED)
    x, y = np.where(mask[:, :, 0] == 1)
    return (x, y)


def rectangle_intersection(a, b):
    """Computes intersection between two rectangles. Allows to check quickly if two bounding boxes overlap."""
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return None
    else:
        return (x, y, w, h)


def rectangle_union(a, b):
    """Computes union between two recangles."""
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return (x, y, w, h)


def iou_by_instance(pred, gt, nb=1):
    """Calculate IoU per Instance for a semantic segmentation mask. Implemented for binary segmentation. pred and gt must be of the same shape (H*W)
    and must both be binary mask. Also returns the IoU and the bounding box associated to the *nb* biggest areas of interest. """
    pred = pred.astype("uint8")  # Necessary conversion for OpenCV
    # Get contours
    _, contours_gt, _ = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, contours_pred, _ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_gt = [i for i in contours_gt if i[:, 0, 0].size > 5]
    contours_pred = [i for i in contours_pred if i[:, 0, 0].size > 5]
    biggest_buildings = np.argsort([i[:, 0, 0].size for i in contours_pred])[-nb:]
    # Initialization
    boxes_gt = list(map(cv2.boundingRect, contours_gt))
    boxes_pred = list(map(cv2.boundingRect, contours_pred))
    iou_instances = []
    empty_instances_pred = []
    empty_instances_gt = []
    mispredicted_round1 = 0  # Check if pred is good with the boxes
    mispredicted_round2 = 0  # Check it wit the shapes.
    iou_big = []
    bb_big = []
    contours = []
    iou_contours = []
    # Main loops
    for i in tqdm(range(len(boxes_pred))):
        for j in range(len(boxes_gt)):
            # First check with the bounding boxes to quickly discard weak predictions

            rect_int = rectangle_intersection(boxes_pred[i], boxes_gt[j])
            rect_un = rectangle_union(boxes_pred[i], boxes_gt[j])

            if rect_int is not None:
                if (rect_int[2] * rect_int[3]) / (rect_un[2] * rect_un[3]) < 0.1:
                    mispredicted_round1 += 1
                else:
                    tic = time()
                    y_max, y_min = rect_un[0] + rect_un[2], rect_un[0]
                    x_max, x_min = rect_un[1] + rect_un[3], rect_un[1]
                    mask_gt, mask_pred = np.zeros((x_max - x_min, y_max - y_min)), np.zeros(
                        (x_max - x_min, y_max - y_min))
                    # Now, check with the original shapes
                    # Step 1: Load the shapes

                    cont_pred, cont_gt = contours_pred[i].copy(), contours_gt[j].copy()
                    cont_pred[:, 0, 0] -= y_min
                    cont_pred[:, 0, 1] -= x_min
                    cont_gt[:, 0, 0] -= y_min
                    cont_gt[:, 0, 1] -= x_min

                    x_gt, y_gt = xy_from_contour(mask_gt, cont_gt)
                    x_pred, y_pred = xy_from_contour(mask_pred, cont_pred)

                    # Step 2: Create masks based on the union of the bounding boxes
                    # and perform dot product on them to calculate the intersection.

                    mask_gt[x_gt, y_gt] = 1
                    mask_pred[x_pred, y_pred] = 1
                    intersection = np.sum(mask_pred * mask_gt)

                    # Step 3: Compute iou and count the number of used areas
                    union = (x_gt.size + x_pred.size - intersection)
                    iou_instances.append(intersection / union)
                    empty_instances_gt.append(j)
                    empty_instances_pred.append(i)
                    if iou_instances[-1] > 0.5:
                        contours.append(contours_pred[i])
                        iou_contours.append(iou_instances[-1])
                    if iou_instances[-1] > 0.5 and i in biggest_buildings:
                        iou_big.append(iou_instances[-1])
                        bb_big.append(rect_un)
                    if iou_instances[-1] < 0.2:
                        # Delete the really weak predictions
                        mispredicted_round2 += 1
                        iou_instances.pop()
                        empty_instances_pred.pop()
                        empty_instances_gt.pop()
    empty_instances_gt = len(contours_gt) - np.unique(empty_instances_gt).size
    empty_instances_pred = len(contours_pred) - np.unique(empty_instances_pred).size
    print("Predicted areas : " + str(len(contours_pred)))
    print("(FP) Wrongly predicted areas : {}".format(empty_instances_pred))
    print("***")
    print("Ground truth areas : " + str(len(contours_gt)))
    print("(FN) Missed GT areas : {}".format(empty_instances_gt))
    print("***")
    print("Weak predictions discarded by boxes/shapes : {}/{}".format(mispredicted_round1,
                                                                      mispredicted_round2))
    print("(TP) Good predictions : " + str(len(iou_instances)))
    print("Instance mean/median/std IoU : {}/{}/{}".format(np.mean(iou_instances), np.median(iou_instances),
                                                           np.std(iou_instances)))
    print("Min : {}, Max : {}".format(np.min(iou_instances), np.max(iou_instances)))
    return (iou_big, bb_big, contours, iou_contours)
