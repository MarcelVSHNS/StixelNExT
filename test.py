import torch
import wandb
import yaml
import time
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import make_scorer

from models.ConvNeXt import ConvNeXt
from dataloader.stixel_multicut import MultiCutStixelData
from dataloader.stixel_multicut_interpreter import StixelNExTInterpreter
from metrics.PrecisionRecall import evaluate_stixels, plot_precision_recall_curve
from utilities.visualization import create_sample_comparison, show_data_pair, plot_roc_curve


# 0.1 Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
# 0.2 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def main():
    fpr_limit = 0.02
    # Data loading
    testing_data = MultiCutStixelData(data_dir=config['data_path'],
                                      phase='testing',
                                      transform=None,
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
                     out_channels=2).to(device)
    weights_file = config['weights_file']
    model.load_state_dict(torch.load("saved_models/" + weights_file))
    print(f'Weights loaded from: {weights_file}')


    # Investigate some selected data
    if config['explore_data']:
        test_features, test_labels, image = next(iter(testing_dataloader))
        # inference
        sample = test_features.to(device)
        output = model(sample)
        output = output.cpu().detach()
        # interpretation
        stixel_reader = StixelNExTInterpreter(detection_threshold=config['pred_threshold'],
                                                   hysteresis_threshold=config['pred_threshold'] - 0.05)
        target_stixel = stixel_reader.extract_stixel_from_prediction(test_labels[0])
        prediction_stixel = stixel_reader.extract_stixel_from_prediction(output[0])

        thresholds = np.linspace(0.01, 1.0, num=100)
        precision_values = []
        recall_values = []

        # Generate precision and recall values at various thresholds
        for iou in thresholds:
            print(f'Threshold: {iou}')
            precision, recall = evaluate_stixels(prediction_stixel, target_stixel, iou_threshold=iou)
            precision_values.append(precision)
            recall_values.append(recall)
            print(f'Precision: {precision}, Recall: {recall}')

        plot_precision_recall_curve(recall_values, precision_values)


    # Create an export of analysed data incl. samples and ROC curve
    if config['logging']['activate']:
        # Init the logger
        # e.g. StixelNExT_ancient-silence-25_epoch-94_loss-0.09816327691078186.pth
        epochs = config['weights']['file'].split('_')[2].split('-')[1]
        checkpoint = config['weights']['file'].split('_')[1]
        wandb_logger = wandb.init(project=config['logging']['project'],
                                  config={
                                      "architecture": type(model).__name__,
                                      "dataset": testing_data.name,
                                      "checkpoint": checkpoint,
                                      "epochs": epochs,
                                      "fpr_limit": fpr_limit
                                  },
                                  tags=["metrics", "testing"]
                                  )


if __name__ == '__main__':
    main()
