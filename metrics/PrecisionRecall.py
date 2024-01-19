import numpy as np
from dataloader.stixel_multicut_interpreter import Stixel, extract_stixels
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def calculate_stixel_iou(pred: Stixel, target: Stixel):
    overlap_start = max(target.top, pred.top)
    overlap_end = min(target.bottom, pred.bottom)
    intersection = max(0, overlap_end - overlap_start)
    union = target.bottom - target.top
    iou = intersection / union if union != 0 else 0
    return iou


def find_best_matches(predicted_stixels, ground_truth_stixels, iou_threshold, rm_used=True):
    hits = 0
    remaining_pred_stixels = set(predicted_stixels)
    # iterate over all GT
    for gt_stixel in ground_truth_stixels:
        matched_pred = None
        # iterate over all Preds
        for pred_stixel in remaining_pred_stixels:
            # iterate over all columns
            if pred_stixel.column == gt_stixel.column:
                # if iou score is high enough; count
                iou_score = calculate_stixel_iou(pred_stixel, gt_stixel)
                if iou_score >= iou_threshold:
                    matched_pred = pred_stixel
                    hits += 1
                    break
        if matched_pred:
            if rm_used:
                remaining_pred_stixels.remove(matched_pred)
    return hits


def evaluate_stixels(predicted_stixels, ground_truth_stixels, iou_threshold):
    """
    Evaluate Stixels with multiple stixels per column using IoU, precision, and recall.
    """
    total_predicted = len(predicted_stixels)
    total_ground_truth = len(ground_truth_stixels)
    hits = find_best_matches(predicted_stixels, ground_truth_stixels, iou_threshold)
    # hits equals True positives (TP)
    # precision = TP / TP + FP
    precision = hits / total_predicted if total_predicted != 0 else 0           # len pred equals True positives + False positives (FP)
    # recall = TP / TP + FN
    recall = hits / total_ground_truth if total_ground_truth != 0 else 0        # len gt equals True positives + False negatives (FN)
    return precision, recall


def plot_precision_recall_curve(recall, precision):
    # Plotting the PR Curve
    f1_scores = [2 * p * r / (p + r) if (p + r) != 0 else 0 for p, r in zip(precision, recall)]
    plt.figure(figsize=(10, 10))
    plt.plot(recall, precision, label='PR Curve')
    plt.plot(recall, f1_scores, label='F1-Score Curve', color='orange', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision / F1-Score')
    plt.title(f'Precision-Recall and F1-Score Curves')
    plt.xlim([0, 1])  # Set x-axis limits
    plt.ylim([0, 1])  # Set y-axis limits
    plt.legend()
    plt.show()

def draw_stixel_on_image_prcurve(image, best_matches, preds, targets, stixel_width=8):
    image = np.array(image.numpy())
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image2 = image.copy()
    for pred in preds:
        cv2.rectangle(image,
                      (pred.column, pred.top),
                      (pred.column + stixel_width, pred.bottom),
                      (255,0,0), 1)

    for targ in targets:
        cv2.rectangle(image2,
                      (targ.column, targ.top),
                      (targ.column + stixel_width, targ.bottom),
                      (0,255,0), 1)

    for match in best_matches.values():
        gt_stixel = match['target_stixel']
        pred_stixel = match['pred_stixel']

        # Berechnung des Überlappungsbereichs
        overlap_top = max(gt_stixel.top, pred_stixel.top)
        overlap_bottom = min(gt_stixel.bottom, pred_stixel.bottom)
        overlap_left = max(gt_stixel.column, pred_stixel.column)
        overlap_right = min(gt_stixel.column + stixel_width, pred_stixel.column + stixel_width)
        """
        # Zeichnen des target_stixel in Grün
        cv2.rectangle(image,
                      (gt_stixel.column, gt_stixel.top),
                      (gt_stixel.column + stixel_width, gt_stixel.bottom),
                      (238,58,140), -1)
        
        cv2.rectangle(image,
                      (gt_stixel.column, gt_stixel.top),
                      (gt_stixel.column + stixel_width, gt_stixel.bottom),
                      (46,139,87), 2)

        # Zeichnen des pred_stixel in Hellgrün

        
        # Zeichnen des Überlappungsbereichs in einer anderen Farbe, z.B. Blau
        if overlap_top < overlap_bottom and overlap_left < overlap_right:
            cv2.rectangle(image,
                          (overlap_left, overlap_top),
                          (overlap_right, overlap_bottom),
                          (0,205,102), -1)  # Blaue Farbe für den Überlappungsbereich

        cv2.rectangle(image,
                      (pred_stixel.column, pred_stixel.top),
                      (pred_stixel.column + stixel_width, pred_stixel.bottom),
                      (118, 238, 198), 2)

        cv2.rectangle(image,
                      (gt_stixel.column, gt_stixel.top),
                      (gt_stixel.column + stixel_width, gt_stixel.bottom),
                      (238,58,140), -1)
        """
    return Image.fromarray(image), Image.fromarray(image2)