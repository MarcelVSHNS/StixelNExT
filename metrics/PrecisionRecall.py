import numpy as np
from dataloader.stixel_multicut_interpreter import Stixel, extract_stixels
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def calculate_stixel_iou(pred: Stixel, target: Stixel):
    """
    Calculate the IoU for two Stixel objects.
    """
    intersection = 0.0
    union = abs(target.top - target.bottom)
    # check if pred overlaps at minimum
    if (target.top <= pred.top <= target.bottom or
        target.top <= pred.bottom <= target.bottom):
        # check if it starts before or after
        if pred.top <= target.top:
            intersection = abs(target.top - pred.bottom) if pred.top < target.top else abs(pred.top - pred.bottom)
            case = 1        # debugging
        # if stixel is below target, remember: row is increasing downwards. pred = 200, target = 400
        elif pred.top > target.top:
            intersection = abs(pred.top - target.bottom) if pred.bottom > target.bottom else abs(pred.top - pred.bottom)
            case = 2        # debugging
    iou = intersection / union if union != 0 else 0
    return iou


def find_best_matches(predicted_stixels, ground_truth_stixels, iou_threshold):
    best_matches = {}  # Store the best match for each ground truth Stixel
    hits = 0
    for gt_stixel in ground_truth_stixels:
        for pred_stixel in predicted_stixels:
            if pred_stixel.column == gt_stixel.column:
                iou_score = calculate_stixel_iou(pred_stixel, gt_stixel)

                # Update the best match if a better one is found
                if iou_score >= iou_threshold and (gt_stixel not in best_matches or iou_score > best_matches[gt_stixel]['iou']):
                    best_matches[gt_stixel] = {'pred_stixel': pred_stixel, 'iou': iou_score, 'target_stixel': gt_stixel}
    # Count hits based on the best matches - TODO: check redundancy
    """matched_predicted = set()
    for gt_stixel, match_info in best_matches.items():
        if match_info['pred_stixel'] not in matched_predicted:
            hits += 1
            matched_predicted.add(match_info['pred_stixel'])
    """
    return best_matches


def evaluate_stixels(predicted_stixels, ground_truth_stixels, iou_threshold):
    """
    Evaluate Stixels with multiple stixels per column using IoU, precision, and recall.
    """
    total_predicted = len(predicted_stixels)
    total_ground_truth = len(ground_truth_stixels)
    best_matches = find_best_matches(predicted_stixels, ground_truth_stixels, iou_threshold)
    hits = len(best_matches)
    # hits equals True positives (TP)
    # precision = TP / TP + FP
    precision = hits / total_predicted if total_predicted != 0 else 0           # len pred equals True positives + False positives (FP)
    # recall = TP / TP + FN
    recall = hits / total_ground_truth if total_ground_truth != 0 else 0        # len gt equals True positives + False negatives (FN)
    return precision, recall, best_matches


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

def draw_stixel_on_image_prcurve(image, best_matches, preds, stixel_width=8):
    image = np.array(image.numpy())
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for pred in preds:
        cv2.rectangle(image,
                      (pred.column, pred.top),
                      (pred.column + stixel_width, pred.bottom),
                      (255,62,150), 1)
    for match in best_matches.values():
        gt_stixel = match['target_stixel']
        pred_stixel = match['pred_stixel']

        # Berechnung des Überlappungsbereichs
        overlap_top = max(gt_stixel.top, pred_stixel.top)
        overlap_bottom = min(gt_stixel.bottom, pred_stixel.bottom)
        overlap_left = max(gt_stixel.column, pred_stixel.column)
        overlap_right = min(gt_stixel.column + stixel_width, pred_stixel.column + stixel_width)

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
    return Image.fromarray(image)
