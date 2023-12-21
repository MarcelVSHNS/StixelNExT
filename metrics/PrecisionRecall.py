import numpy as np
from dataloader.stixel_multicut_interpreter import Stixel, extract_stixels
from collections import defaultdict
import matplotlib.pyplot as plt


def calculate_stixel_iou(pred: Stixel, target: Stixel):
    """
    Calculate the IoU for two Stixel objects.
    """
    intersection = 0.0
    union = abs(target.top - target.bottom)
    if (target.top <= pred.top <= target.bottom or
        target.top <= pred.bottom <= target.bottom):
        if pred.top <= target.top:
            # case 1
            intersection = abs(target.top - pred.bottom) if pred.top < target.top else abs(pred.top - pred.bottom)
            case = 1
        elif pred.top > target.top:
            # case 2
            intersection = abs(pred.top - target.bottom) if pred.bottom > target.bottom else abs(pred.top - pred.bottom)
            case = 2
    return intersection / union if union != 0 else 0


def find_best_matches(predicted_stixels, ground_truth_stixels, iou_threshold):
    best_matches = {}  # Store the best match for each ground truth Stixel
    hits = 0
    for gt_stixel in ground_truth_stixels:
        for pred_stixel in predicted_stixels:
            if pred_stixel.column == gt_stixel.column:
                iou_score = calculate_stixel_iou(pred_stixel, gt_stixel)

                # Update the best match if a better one is found
                if iou_score >= iou_threshold and (gt_stixel not in best_matches or iou_score > best_matches[gt_stixel]['iou']):
                    best_matches[gt_stixel] = {'pred_stixel': pred_stixel, 'iou': iou_score}
    # Count hits based on the best matches - TODO: check redundancy
    """matched_predicted = set()
    for gt_stixel, match_info in best_matches.items():
        if match_info['pred_stixel'] not in matched_predicted:
            hits += 1
            matched_predicted.add(match_info['pred_stixel'])
    """
    return len(best_matches)    # hits


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
    plt.figure()
    plt.plot(recall, precision, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0, 1])  # Set x-axis limits
    plt.ylim([0, 1])  # Set y-axis limits
    plt.legend()
    plt.show()
