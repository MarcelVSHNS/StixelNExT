import yaml
# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

import os
import torch
import wandb
import time
import numpy as np
from torch.utils.data import DataLoader
from models.ConvNeXt import ConvNeXt
from dataloader.stixel_multicut import MultiCutStixelData
from dataloader.stixel_multicut_interpreter import StixelNExTInterpreter, show_pred_heatmap
from metrics.PrecisionRecall import evaluate_stixels, plot_precision_recall_curve, draw_stixel_on_image_prcurve
if config['dataset'] == "kitti":
    from dataloader.stixel_multicut import feature_transform_resize as feature_transform
else:
    feature_transform = None

# 0.2 Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Data loading
    dataset_dir = os.path.join(config['data_path'], config['dataset'])
    testing_data = MultiCutStixelData(data_dir=dataset_dir,
                                      phase='testing',
                                      transform=feature_transform,
                                      target_transform=None,
                                      return_original_image=True)
    testing_dataloader = DataLoader(testing_data, batch_size=config['batch_size'],
                                num_workers=config['resources']['test_worker'], pin_memory=True, shuffle=True,
                                drop_last=True)

    # Set up the Model
    model = ConvNeXt(stem_features=config['nn']['stem_features'],
                     depths=config['nn']['depths'],
                     widths=config['nn']['widths'],
                     drop_p=config['nn']['drop_p'],
                     target_height=int(testing_data.img_size['height'] / config['grid_step']),
                     target_width=int(testing_data.img_size['width'] / config['grid_step']),
                     out_channels=2).to(device)

    weights_file = config['weights_file']
    checkpoint = os.path.splitext(weights_file)[0]  # checkpoint without ending
    run = checkpoint.split('_')[1]
    model.load_state_dict(torch.load(os.path.join("saved_models", run, weights_file)))
    print(f'Weights loaded from: {weights_file}')

    # Investigate some selected data
    if config['explore_data']:
        iou_pick = 50   # in %
        test_features, test_labels, image = testing_data[101]
        # inference

        sample = test_features.unsqueeze(0).to(device)
        output = model(sample)
        output = output.cpu().detach()
        output = output.squeeze()
        # interpretation
        # show_pred_heatmap(image, output)
        # show_pred_heatmap(image, output, map=1)
        stixel_reader = StixelNExTInterpreter(detection_threshold=config['pred_threshold'],
                                                   hysteresis_threshold=config['pred_threshold'] - 0.05)
        target_stixel = stixel_reader.extract_stixel_from_prediction(test_labels)
        stixel_reader.show_stixel(image, color=[0, 255, 0])
        prediction_stixel = stixel_reader.extract_stixel_from_prediction(output)  # compare: output/ test_labels
        stixel_reader.show_stixel(image, color=[255, 0, 0])
        precision, recall = evaluate_stixels(prediction_stixel, target_stixel, iou_threshold=0.5)


        thresholds = np.linspace(0.1, 1.0, num=10)
        precision_values = []
        recall_values = []
        matches_collections = []

        # Generate precision and recall values at various thresholds
        for iou in thresholds:
            print(f'Threshold: {iou}')
            precision, recall, best_matches = evaluate_stixels(prediction_stixel, target_stixel, iou_threshold=iou)
            precision_values.append(precision)
            recall_values.append(recall)
            matches_collections.append(best_matches)
            print(f'Precision: {precision}, Recall: {recall}')
        img, img2 = draw_stixel_on_image_prcurve(image,matches_collections[iou_pick], prediction_stixel, target_stixel)
        img.show()
        img2.show()
        plot_precision_recall_curve(recall_values, precision_values)


    # Create an export of analysed data incl. samples and ROC curve
    if config['logging']['activate']:
        # Init the logger
        # e.g. StixelNExT_ancient-silence-25_epoch-94_loss-0.09816327691078186.pth
        epochs = config['weights_file'].split('_')[2].split('-')[1]
        checkpoint = config['weights_file'].split('_')[1]
        wandb_logger = wandb.init(project=config['logging']['project'],
                                  config={
                                      "architecture": type(model).__name__,
                                      "dataset": testing_data.name,
                                      "checkpoint": checkpoint,
                                      "epochs": epochs
                                  },
                                  tags=["metrics", "testing"]
                                  )

        stixel_reader = StixelNExTInterpreter(detection_threshold=config['pred_threshold'],
                                              hysteresis_threshold=config['pred_threshold'] - 0.05)
        thresholds = np.linspace(0.1, 1.0, num=10)
        prec_ious = []
        recall_ious = []
        for iou in thresholds:
            prec_batches = []
            recall_batches = []
            for batch_idx, (samples, targets, images) in enumerate(testing_dataloader):
                samples = samples.to(device)
                start = time.process_time_ns()
                output = model(samples)
                t_infer = time.process_time_ns() - start
                # fetch data from GPU
                output = output.cpu().detach()
                prec_samples = []
                recall_samples = []
                for i in range(output.shape[0]):
                    target_stixel = stixel_reader.extract_stixel_from_prediction(targets[i])
                    prediction_stixel = stixel_reader.extract_stixel_from_prediction(output[i])
                    precision, recall, best_matches = evaluate_stixels(prediction_stixel, target_stixel, iou_threshold=iou)
                    prec_samples.append(precision)
                    recall_samples.append(recall)
                prec_batches.append(sum(prec_samples)/ len(prec_samples))
                recall_batches.append(sum(recall_samples)/ len(recall_samples))

            prec_ious.append(sum(prec_batches)/ len(prec_batches))
            recall_ious.append(sum(recall_batches) / len(recall_batches))
        f1_scores = [2 * p * r / (p + r) if (p + r) != 0 else 0 for p, r in zip(prec_ious, recall_ious)]
        data = [[x, y] for (x, y) in zip(recall_ious, prec_ious)]
        table = wandb.Table(data=data, columns=["Recall", "Precision"])
        wandb_logger.log({"PR curve": wandb.plot.line(table, "Recall", "Precision",
                                                       title=f"Precision-Recall over {testing_data.__len__()} samples")})
        wandb_logger.log({"F1": sum(f1_scores)/ len(f1_scores)})


if __name__ == '__main__':
    main()
